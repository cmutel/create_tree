"""
Algorithm that converts "matrix LCA" (network) product systems into 
acyclic rooted trees.
Built for use within the Brightway2 LCA framework (https://brightwaylca.org/).

Instantiating a Tree (`tree = Tree(demand)` object generates a "network LCA" object that 
is used in the construction of the rooted tree.
The actual construction of the rooted tree is done by calling `tree.traverse`
The resulting tree product system can b    - The root is the unit process providing the functional unit.
    - All branches (i.e. internal nodes) are unit processes whose 
        cradle-to-gate contributions are greater than a specified threshold 
        (the "cutoff") for a given list of methods.
    - All other unit processes are considered leaves.
    - Leaves can be:
        * "entry points" to a disaggregated LCI database
        * Accumulated datasets from an aggregated LCI database
        * Datasets produced on the fly containing only LCIA scores
    - The resulting branch and leaf nodes can be writen to respective databasese written to databases using `tree.write`

Useful for reducing the number of parameters in an LCA model and for investigating paths.

Developed by Pascal Lesage and Chris Mutel.
"""

from brightway2 import *
from backend import ProcessedBackend
import collections
import copy
import itertools
import numpy as np


class CounterDict(object):
    """Class that counts each time a value is looked up."""
    def __init__(self):
        self.d = collections.defaultdict(itertools.count) 
        #defaultdict defaults to something when k doesn't exist. In this case, it defaults to an itertools.count

    def __getitem__(self, key):
        """Returns a k:v in d equal to key:the number of times that key has come up.
           Used for creating IDs"""
        return next(self.d[key])

        
class Tree(object):
    """Rooted tree representations of product systems."""
    def __init__(self, demand):
        self.demand = demand
        self.network_lca = LCA(demand)
        self.network_lca.lci(factorize=True)

    def get_C_matrices_and_scores(self):
        """For a list of LCIA methods, create dictionaries containing:
            1) matrices of characterization factors; and
            2) cradle-to-gate LCIA results"""
        self.c_matrices = {}
        self.scores = {}
        for method in self.methods:
            self.network_lca.switch_method(method)
            self.network_lca.lcia_calculation()
            self.scores[method] = self.network_lca.characterized_inventory.sum()
            self.c_matrices[method] = self.network_lca.characterization_matrix

    def initialize_deque(self, lca, counter_dict):
        """Use deque to get high performance additions & deletions 
        (https://docs.python.org/2/library/collections.html#collections.deque)
        We use `append` and `popleft` to get breadth-first search.
        Initializing the deque adds the root activity"""
        deque = collections.deque()
        for activity, amount in lca.demand.items():
            new_code = ("{}-{}".format(activity[1], counter_dict[activity[1]]))
            # Calculate cradle-to-gate results
            root_c2g = {method: self.scores[method]\
                        for method in self.methods\
                        }     
            col_index = self.network_lca.activity_dict[activity]
            scale_value = lca.technosphere_matrix[col_index, col_index]
            root_g2g = {method: (amount/scale_value)*(self.c_matrices[method]*lca.biosphere_matrix[:, col_index]).sum()\
                        for method in self.methods}  # Gate to gate results
            node_type = "root"
            deque.append((activity, amount, new_code, node_type, root_g2g, root_c2g))
            return deque

    def relabel_exchange(self, exc, target_database, new_code, relabel_input=False):
        if relabel_input:
            exc['input'] = (target_database, new_code)
        exc['output'] = (target_database, new_code)
        return exc
    
    def traverse(self, methodList, tree_database, cutoff = 0.1, store_leaves_as_scores = True):
        """Function that traverses the tree and distinguishes between branches and leaves.
            - methodList: List of LCIA methods, as identified in Brightway.
            - tree_database: Name for the new database that will store the tree.
            - cutoff: Threshold for "remaining" cradle-to-gate impact that distinguishes branches from leaves.
            - store_leaves_as_scores: If True, will store aggregated LCIA results. 
                                      If False, links to background database are kept."""

        self.network_lca.redo_lci(self.demand)
        self.methods = methodList
        self.cutoff = cutoff
        self.tree_database = tree_database
        self.store_leaves_as_scores = store_leaves_as_scores
        self.list_leafs = []
        self.list_branches = []
        self.get_C_matrices_and_scores()
        counter_dict = CounterDict() # Dictionary to count how many times we have seen an activity
        deque = self.initialize_deque(self.network_lca, counter_dict)

        # Get dictionary of matrix indices to activity keys
        reverse_activity, reverse_product, _ = self.network_lca.reverse_dict()
        self.listTreeNodeDatasets = []

        while deque:  # As long as there are activities to process in the deque
            demand, amount, new_code, node_type, g2g_contribution, c2g_contribution = deque.popleft()
            col_index = self.network_lca.activity_dict[demand]

            # Get activity and underlying dataset
            activity = get_activity(demand)
            dataset = copy.deepcopy(activity._data)
            dataset.update({
                'database': self.tree_database,
                'old database': dataset['database'],
                'node type': node_type, 
                'old code': dataset['code'],
                'code': new_code,
                'exchanges': [],
                'gate-to-gate contribution': g2g_contribution,
                'cradle-to-gate contribution': c2g_contribution
            })
            if dataset['node type'] == "root":
                self.root = (self.tree_database, new_code)
            self.list_branches.append(activity)

            # Add all production exchanges
            for exc in activity.production():
                exc_dataset = copy.deepcopy(exc._data)
                if exc_dataset['input'] in [exc['input'] for exc in activity.technosphere()]:
                    exc_dataset['amount'] += -[exc['amount'] for exc in activity.technosphere() if exc['input'] == exc['output']][0]
                self.relabel_exchange(exc_dataset, self.tree_database, new_code, True)

                dataset['exchanges'].append(exc_dataset)

            # Add all biosphere exchanges
            for exc in activity.biosphere():
                exc_dataset = copy.deepcopy(exc._data)
                self.relabel_exchange(exc_dataset, self.tree_database, new_code)
                dataset['exchanges'].append(exc_dataset)

            # We assume that our database has only single-output activities.
            # However, we can't assume that this activity produces one unit
            # of its reference product.
            scale_value = self.network_lca.technosphere_matrix[col_index, col_index]
            assert scale_value, "Can't process 0 production of reference product"

            # Lookup exchanges in technosphere matrix. First find our activity's column
            column = self.network_lca.technosphere_matrix[:, col_index].tocoo()
            exchanges = [(
                # Activity key
                reverse_product[int(column.row[i])],
                # Net amount of this exchange.
                # Multiply by -1 because inputs are negative in A matrix.
                float(-1 *
                      amount *       # Amount of 'parent' activity
                      column.data[i] /  # Exchange amount
                      scale_value)
                ) for i in range(column.row.shape[0])
                # Skip production exchange
                if column.row[i] != col_index
            ]

            links_to_tree_nodes_this_loop = {}

            for child_activity, child_amount in exchanges:
                child_code = "{}-{}".format(child_activity[1], str(counter_dict[child_activity[1]]))
                self.network_lca.redo_lci({child_activity: child_amount})
                child_characterized_inventories = {method: (self.c_matrices[method] * self.network_lca.inventory) for method in self.methods}
                child_col_index = self.network_lca.activity_dict[child_activity]
                g2g_contribution = {}
                for method, characterized_LCI in child_characterized_inventories.items():
                    g2g_contribution[method] = characterized_LCI[:,child_col_index].sum()
                
                c2g_contribution = {method: characterized_LCI.sum()\
                                        for method, characterized_LCI\
                                        in child_characterized_inventories.items()}
                relative_contributions = [c2g_contribution[method]/self.scores[method] for method in self.methods]
                
                if any (contribution>self.cutoff for contribution in relative_contributions):
                    

                    # Store mapping from activity key to new code
                    
                    links_to_tree_nodes_this_loop[child_activity] = child_code
                    deque.append((child_activity, child_amount, child_code, "branch", g2g_contribution, c2g_contribution))
                else:
                    if not self.store_leaves_as_scores:
                        self.list_leafs.append(child_activity)
                    else:
                        self.list_leafs.append((self.tree_database, child_code))
                        links_to_tree_nodes_this_loop[child_activity] = child_code
                        leaf_act = get_activity((child_activity))  
                        leaf_dataset = copy.deepcopy(leaf_act._data)
                        leaf_dataset.update({
                            'database': self.tree_database,
                            'old database': leaf_act['database'],
                            'old code': leaf_act['code'],
                            'code': child_code,
                            'node type': 'leaf',
                            'cradle-to-gate contribution': c2g_contribution,
                            'exchanges': []
                        })

                        for exc in leaf_act.production():
                            exc_dataset = copy.deepcopy(exc._data)
                            exc_dataset['amount'] = child_amount
                            self.relabel_exchange(exc_dataset, self.tree_database, child_code, True)
                            leaf_dataset['exchanges'].append(exc_dataset)
                        

                        for method in self.methods:
                            leaf_dataset['exchanges'].append({\
                                'input': ('biosphere3', "unit impact - {}".format(Method(method).get_abbreviation())),\
                                'amount': c2g_contribution[method],\
                                'output': child_code,\
                                'type': 'biosphere'})
                        self.listTreeNodeDatasets.append(leaf_dataset)
                                                
                # Add technosphere exchanges to dataset
            for exc in activity.technosphere(): 
                exc_dataset = copy.deepcopy(exc._data)
                # Change input code if needed
                if exc['input'] == exc['output']:
                    pass
                else:
                    try:
                        exc_dataset['input'] = (self.tree_database, links_to_tree_nodes_this_loop[exc_dataset['input']])
                    except KeyError:
                        pass
                    self.relabel_exchange(exc_dataset, self.tree_database, new_code)
            
                if not exc['input'] == exc['output']:
                    dataset['exchanges'].append(exc_dataset)
            self.listTreeNodeDatasets.append(dataset)

    def write_new_data(self, new_name, node_data, only_processed=True):
        if only_processed:
            database = ProcessedBackend(new_name)
        else:
            database = Database(new_name)
        database.write({(o['database'], o['code']): o for o in node_data})
        return database

    def write_tree(self):
        assert len(self.listTreeNodeDatasets) > 0, "The tree has not yet been created. Call `tree.traverse() first."
        projects.read_only = False        
        self.write_new_data(self.tree_database, self.listTreeNodeDatasets)