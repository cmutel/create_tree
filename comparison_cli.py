# -*- coding: utf-8 -*-
from brightway2 import *
from bw2data.utils import random_string
from create_tree import Tree # Code used to generate "branch and leaf" systems
from math import ceil
from time import sleep
import click
import multiprocessing as mp
import numpy as np
import os
import pickle
import time


DEFAULT_METHODS = [
    ('ReCiPe Midpoint (H)', 'metal depletion', 'MDP'),
    ('ReCiPe Midpoint (H)', 'water depletion', 'WDP'),
    ('ReCiPe Midpoint (H)', 'urban land occupation', 'ULOP'),
    ('ReCiPe Midpoint (H)', 'terrestrial ecotoxicity', 'TETPinf'),
    ('ReCiPe Midpoint (H)', 'marine eutrophication', 'MEP'),
    ('ReCiPe Midpoint (H)', 'freshwater ecotoxicity', 'FETPinf'),
    ('ReCiPe Midpoint (H)', 'photochemical oxidant formation', 'POFP'),
    ('ReCiPe Midpoint (H)', 'particulate matter formation', 'PMFP'),
    ('ReCiPe Midpoint (H)', 'ionising radiation', 'IRP_HE'),
    ('ReCiPe Midpoint (H)', 'marine ecotoxicity', 'METPinf'),
    ('ReCiPe Midpoint (H)', 'human toxicity', 'HTPinf'),
    ('ReCiPe Midpoint (H)', 'ozone depletion', 'ODPinf'),
    ('ReCiPe Midpoint (H)', 'agricultural land occupation', 'ALOP'),
    ('ReCiPe Midpoint (H)', 'freshwater eutrophication', 'FEP'),
    ('ReCiPe Midpoint (H)', 'fossil depletion', 'FDP'),
    ('ReCiPe Midpoint (H)', 'natural land transformation', 'NLTP'),
    ('ReCiPe Midpoint (H)', 'climate change', 'GWP100')
]


def chunks(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]


@click.command()
@click.option('--version', help='Ecoinvent version (2 or 3)', type=int)
@click.option('--methods', default=None, help='JSON file with list of methods')
@click.option('--cutoff', help='Cutoff value in (0, 1)', type=float)
@click.option('--iterations', default=1000, help='Number of Monte Carlo iterations', type=int)
@click.option('--cpus', default=mp.cpu_count(), help='Number of used CPU cores', type=int)
@click.option('--output', help='Output directory path',
              default=r'E:\test\network_tree_comparison\raw')
def main(version, methods, cutoff, cpus, output, iterations):
    """Module used to compare the uncertainty of networks and trees.

    Uses Brightway2 framework and the Tree class. For a list of
    activities, Monte Carlo is carried out both on the network
    representation of the product system and on "rooted tree"
    versions.

    """
    if version not in (2,3):
        print("Need valid version number")
        return

    base_project = "2.2 tree base" if version == 2 else "3.3 tree base"

    if methods is None:
        methods = DEFAULT_METHODS
    else:
        raise NotImplemented("Not done yet")

    if not 0 < cutoff < 1:
        print("Invalid cutoff value")
        return

    if not output or not os.path.isdir(output):
        print("This is not a valid path: {}".format(output))
        return

    assert cpus
    assert iterations

    projects.set_current(base_project)

    ecoinvent = Database('ecoinvent')

    # In case you need to stop and restart program
    # Results are stored by the activity code + ".pickle" - subtract
    # even characters to get the code back.
    already_treated = [file[:-7] for file in os.listdir(output)
                       if file[-7:] == '.pickle']
    to_treat = [act.key for act in ecoinvent
                if act.key[1] not in already_treated][:20]

    activity_sublists = chunks(to_treat, ceil( len(to_treat) / cpus ))

    workers = []
    for index, sub_list in enumerate(activity_sublists):
        j = mp.Process(target=tree_worker,
                       args=(base_project,
                             sub_list,
                             cutoff,
                             methods,
                             iterations,
                             output,
                             index
                            )
                       )
        workers.append(j)
    for w in workers:
        print("starting worker {}".format(w))
        w.start()
        sleep(3)


def get_characterization_matrices(demand, list_of_methods):
    characterization_matrices = {}
    sacrificial_LCA = LCA({demand:1})
    sacrificial_LCA.lci()
    for method in list_of_methods:
        sacrificial_LCA.switch_method(method)
        characterization_matrices[method] = sacrificial_LCA.characterization_matrix.copy()
    return characterization_matrices


def tree_worker(base_project, activities, cutoff,
                list_methods, iterations, output_fp, worker_id):
    """Function that generates Monte Carlo results for network and tree
    representations of product systems."""

    # Create new project for this worker. Will be deleted later.
    projects.set_current(base_project)
    new_project_name = "temp project {}".format(random_string())
    projects.copy_project(new_project_name)

    ecoinvent = Database('ecoinvent')

    # Generate matrices with characterization factors. This is much faster than
    # using "switch_method" during Monte Carlo.
    characterization_matrices = get_characterization_matrices(activities[0], list_methods)

    for i, act_key in enumerate(activities):
        act = get_activity(act_key)

        # Create dict with tree specifications (in terms of cut-off)
        model_types = ['network', 'deterministic', 'tree-{}'.format(cutoff)]

        # Generate results collector
        results = {method: {} for method in list_methods}
        for method in list_methods:
            for model_type in model_types:
                if model_type != 'deterministic':
                    results[method][model_type] = np.zeros(iterations)

        # Generate metadata collector
        tree_metadata = {}

        # Deterministic
        lca = LCA({act:1})
        lca.lci()
        for method, cfs in characterization_matrices.items():
            results[method]['deterministic'] = (cfs * lca.inventory).sum()

        MC = MonteCarloLCA({act:1})
        for iteration in range(iterations):
            s = next(MC) # s contains supply array
            lci = MC.biosphere_matrix * s
            for method, cfs in characterization_matrices.items():
                results[method]['network'][iteration] = (cfs*lci).sum()

        # Tree LCAs
        tree = Tree({act:1})
        tree.traverse(list_methods,
                      'temp-{}'.format(act['code']),
                      cutoff=cutoff,
                      store_leaves_as_scores=False)

        # Store some information on tree
        tree_metadata = {}
        tree_metadata['number of branches'] = len(tree.list_branches)
        tree_metadata['number of leaves'] = len(tree.list_leafs)
        tree_metadata['branch_contribution'] = {}

        for method in tree.methods:
            tree_metadata['branch_contribution'][method] = 0
            for node in tree.listTreeNodeDatasets:
                if (node['node type'] == 'branch' or
                    node['node type'] == 'root'):
                    tree_metadata['branch_contribution'][method] += (
                        node['gate-to-gate contribution'][method] /
                            tree.scores[method]
                    )

        tree.write_tree()

        # Do Monte Carlo
        MC = MonteCarloLCA({tree.root:1})
        for iteration in range(iterations):
            s = next(MC)
            lci = MC.biosphere_matrix * s
            for method, cfs in characterization_matrices.items():
                results[method]['tree-{}'.format(cutoff)][iteration] = (cfs * lci).sum()

        #Dump results to disk
        with open(os.path.join(output_fp, "{}.pickle".format(act['code'])),"wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(output_fp,
                               "_metadata_{}.pickle".format(act['code'])),
                 "wb") as f:
            pickle.dump(tree_metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    projects.delete_project(new_project_name, True)

if __name__ == "__main__":
    main()
