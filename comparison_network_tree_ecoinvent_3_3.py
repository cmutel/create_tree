"""Module used to compare the uncertainty of networks and trees.

Uses Brightway2 framework and the Tree class. For a list of
activities (by default, all of ecoinvent v3.3), Monte Carlo is carried out
both on the network representation of the product system and on "rooted
tree" versions.
Current version can run code for a list of cut-off criteria.
Current version will use *all* ReCiPe methods to evaluate whether upstream
impacts are below cut-off.
"""

import os
import pickle
from brightway2 import *
import numpy as np
import time
import multiprocessing as mp
from math import ceil
from time import sleep

from create_tree import Tree # Code used to generate "branch and leaf" systems
import bw_utils

def MC_network_and_trees(project_name, act_key_list, tree_cut_off_list,
                         list_methods, iterations, dump_fp, worker):
    """Function that generates Monte Carlo results for network and tree
    representations of product systems. It is presently configured for use
    with ecoinvent v3.3 database.
    """
    
    # Generate project specifically for this job. The project can (and probably
    # should) be deleted after this exercise.
    projects.set_current(project_name)
    projects.read_only = False

    # Import data in the throw-away project.
    bw2setup()
    # CHRIS: Please change filepath to location of ecoinvent v3.3 files
    if 'ecoinvent 3.3 cut-off' not in databases:
        ei33cu = SingleOutputEcospold2Importer(
            r"E:\ecoinvent_3_3_cut_off\datasets",
            'ecoinvent 3.3 cut-off')
        ei33cu.apply_strategies()
        ei33cu.write_database()
    else:
        print("ecoinvent 3.3 cut-off already imported")
    ei33cu = Database('ecoinvent 3.3 cut-off')
    
    # Generate matrices with characterization factors. This is much faster than
    # using "switch_method" during Monte Carlo.
    C_matrices = bw_utils.get_C_matrices(ei33cu.random(), list_methods)
    
    for i, act_key in enumerate(act_key_list):
        act = Database(act_key[0]).get(act_key[1])
        
        # Create dict with tree specifications (in terms of cut-off)
        tree_dict = {'tree_'+str(cutoff): cutoff \
                        for cutoff in tree_cut_off_list}
        # Create dict to collect results per model type.
        model_types = list(tree_dict.keys())
        model_types.append('network')
        model_types.append('deterministic')
        
        # Generate results collector
        results = {}
        for method in C_matrices:
            results[method] = {}
            for model_type in model_types:
                if model_type != 'deterministic':
                    results[method][model_type] = np.zeros(iterations)
        # Generate metadata collector
        tree_metadata = {}
            
        # Deterministic
        try:
            lca = LCA({act:1})
            lca.lci()
            for method, cfs in C_matrices.items():
                results[method]['deterministic'] = (cfs*lca.inventory).sum()
        except:
            print("{} failed during deterministic!".format(act))
        # Network LCA
        print('start work on network')
        try:
            MC = MonteCarloLCA({act:1})
            for iteration in range(iterations):
                s = next(MC) # s contains supply array
                lci = MC.biosphere_matrix * s
                for method, cfs in C_matrices.items():
                    results[method]['network'][iteration] = (cfs*lci).sum()
        except:
            print("{} failed during network!".format(act))
        # Tree LCAs
        for cut_off_tree in tree_dict.keys():
            try:
                # Create and traverse tree
                tree = Tree({act:1})
                tree.traverse([method for method in C_matrices.keys()],
                              'temp_tree',
                              cutoff=tree_dict[cut_off_tree],
                              store_leaves_as_scores=False)
                # Store some information on tree
                tree_metadata[cut_off_tree] = {}
                tree_metadata[cut_off_tree]['number of branches'] = len(
                                                                tree.list_branches)
                tree_metadata[cut_off_tree]['number of leaves'] = len(
                                                                tree.list_leafs)
                tree_metadata[cut_off_tree]['branch_contribution'] = {}

                for method in tree.methods:
                    tree_metadata[cut_off_tree]['branch_contribution'][method] = 0
                    for node in tree.listTreeNodeDatasets:
                        if (node['node type'] == 'branch' or
                            node['node type'] == 'root'):
                            tree_metadata[cut_off_tree]['branch_contribution'][method] +=\
                                node['gate-to-gate contribution'][method]/\
                                tree.scores[method]
                # Write tree to database
                projects.read_only = False
                tree.write_tree()
                # Do Monte Carlo
                MC = MonteCarloLCA({tree.root:1})
                for iteration in range(iterations):
                    s = next(MC)
                    lci = MC.biosphere_matrix * s
                    for method, cfs in C_matrices.items():
                        results[method][cut_off_tree][iteration] = (cfs * lci).sum()
                # Delete temporary database
                Database('temp_tree').delete()
                Database('temp_tree').deregister()
            except:
                print("{} failed during tree!".format(act))
        #Dump results to disk
        with open(os.path.join(dump_fp, "{}.pickle".format(act_key[1])),"wb") as f:
            pickle.dump(results, f)
        with open(os.path.join(dump_fp,
                               "_metadata_{}.pickle".format(act_key[1])),
                 "wb") as f:
            pickle.dump(tree_metadata, f)

def chunks(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]

if __name__ == '__main__':
    projects.set_current('compare uncertainties networks and trees_ei33cu')
    bw2setup()
    # CHRIS: change filepath to location of ecoinvent v3.3 files
    if 'ecoinvent 3.3 cut-off' not in databases:
        ei33cu = SingleOutputEcospold2Importer(
            r"E:\ecoinvent_3_3_cut_off\datasets",
            'ecoinvent 3.3 cut-off')
        ei33cu.apply_strategies()
        ei33cu.write_database()
    else:
        print("ecoinvent 3.3 cut-off already imported")
    ei33cu = Database('ecoinvent 3.3 cut-off')
    list_methods = [
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
    
    tree_cut_off_list = [0.1] # Cut-off criteria 
    list_of_activities_keys = [act.key for act in Database('ecoinvent 3.3 cut-off')]
    # CHRIS: Please change filepath to location where you want to dump results    
    output_dir_fp = r'E:\test\network_tree_comparison\raw'

    # In case you need to stop and restart program
    already_treated = [file[:-7] for file in os.listdir(output_dir_fp)]
    to_treat = [file for file in list_of_activities_keys\
                if file[1] not in already_treated]

    CPUS = 1#mp.cpu_count()
    activity_sublists = chunks(to_treat, ceil(len(to_treat)/CPUS))

    projects.set_current('default')

    workers = []
    for i, sub_list in enumerate(activity_sublists):            
        j = mp.Process(target=MC_network_and_trees, 
                       args=('compare uncertainties networks and trees-{}'.format(i),
                             sub_list,
                             tree_cut_off_list,
                             list_methods,
                             2,
                             output_dir_fp,
                             i
                            )
                       )

        workers.append(j)
    for w in workers:
        print("starting worker {}".format(w))
        w.start()
        sleep(3)