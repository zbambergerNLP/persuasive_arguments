import networkx as nx
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from collections import Counter
import pdb
from parser import Entity, Attribute, Relations

class GraphFeatures:
    """
    :param graph: the graph of interest

    Attributes:
        roots                   nodes with in-degree = 0
        nroots                  the number of root nodes

        leaves                  nodes with out-degree = 0
        nleaves                 the number of leaf nodes

        root_type_counts        entity type counts over all root nodes
        leaf_type_counts        entity type counts over all leaf nodes
        node_type_counts        entity type counts over all nodes
        edge_type_counts        relation type counts over all edges

        all_paths               all root to leaf paths
        all_paths_type_counts   entity type counts over all root to leaf paths
        avg_len_path            average root to leaf path length
        max_len_path            maximum root to leaf path length

        nchildren_counts        successor counts for all nodes
        avg_children_counts     average successor counts for all nodes
        max_children_counts     maximum successor counts for all nodes

        entity_relations_mat    relation counts across entity types,
                                len(Entity.TYPES) x len(Entity.TYPES) x
                                len(Relations.TYPES)

        nodes_dist_k_from_root  nodes k distance from root
        nnodes_dist_k_from_root count of nodes k distance from root
        nodes_dist_k_from_leaf  nodes k distance from leaf
        nnodes_dist_k_from_leaf count of nodes k distance from leaf

    """
    def __init__(self, graph, k):
        self.graph = graph
        self.roots =  [n for n,d in graph.in_degree() if d==0]
        self.nroots =  len(self.roots)

        self.leaves = [n for n,d in graph.out_degree() if d==0]
        self.nleaves = len(self.leaves)

        # note that nodes may be both a root and leaf
        self.root_type_counts = Counter({t:0 for t in Entity.TYPES})
        for n in self.roots:
            self.root_type_counts[graph.nodes[n]['obj'].type] += 1

        self.leaf_type_counts = Counter({t:0 for t in Entity.TYPES})
        for n in self.leaves:
            self.leaf_type_counts[graph.nodes[n]['obj'].type] += 1

        self.node_type_counts = Counter({t:0 for t in Entity.TYPES})
        for n in self.graph.nodes():
            self.node_type_counts[graph.nodes[n]['obj'].type] += 1

        self.edge_type_counts = Counter({t.lower():0 for t in Relations.TYPES})
        for e in self.graph.edges():
            if type(graph.edges()[e]['obj']) == Relations:
                self.edge_type_counts[graph.edges()[e]['obj'].type] += 1

        self.all_paths = []
        self.all_paths_type_counts = Counter({t:0 for t in Entity.TYPES})
        for root in self.roots:
            for leaf in self.leaves:
                for path in nx.all_simple_paths(graph, root, leaf):
                    self.all_paths.append(path)
                    for n in path:
                        self.all_paths_type_counts[graph.nodes[n]['obj'].type] += 1

        n_all_paths = len(self.all_paths)
        for t in self.all_paths_type_counts:
            self.all_paths_type_counts[t] /= n_all_paths

        len_paths = [len(p) for p in self.all_paths]
        self.avg_len_path = sum(len_paths) / len(len_paths)
        self.max_len_path = max(len_paths)

        self.nchildren_counts = [len([s for s in graph.successors(n)]) for n in graph.nodes()]
        self.avg_children_counts = np.mean(self.nchildren_counts)
        self.max_children_counts = np.max(self.nchildren_counts)

        # construct matrix to store relations counts between entities
        n_entity_types = len(Entity.TYPES)
        n_relations_types = len(Relations.TYPES)

        inv_idx_entity_types = {}
        for t in Entity.TYPES:
            inv_idx_entity_types[t] = Entity.TYPES.index(t)
        inv_idx_relations_types = {}
        for t in Relations.TYPES:
            inv_idx_relations_types[t] = Relations.TYPES.index(t)

        self.entity_relations_mat = np.zeros((n_entity_types,n_entity_types,n_relations_types))

        for e in self.graph.edges():
            if type(graph.edges()[e]['obj']) == Relations:
                (i,j) = e
                i_entity_type = graph.nodes[i]['obj'].type
                j_entity_type = graph.nodes[j]['obj'].type

                mat_idx_i = inv_idx_entity_types[i_entity_type]
                mat_idx_j = inv_idx_entity_types[j_entity_type]

                e_relations_type = graph.edges()[e]['obj'].type
                mat_idx_r = inv_idx_relations_types[e_relations_type]

                self.entity_relations_mat[mat_idx_i, mat_idx_j, mat_idx_r] += 1

        # compute nodes k distance from root
        # set that contains all nodes distance k from a root
        self.nodes_dist_k_from_root = set()
        for root in self.roots:
            # compute the shortest path lengths from this root to all reachable nodes
            paths = nx.single_source_shortest_path_length(graph, root, cutoff=k)
            for key in paths:
                self.nodes_dist_k_from_root.add(key)

        self.nnodes_dist_k_from_root = len(self.nodes_dist_k_from_root)

        # compute nodes k distance from leaf
        # set that contains all nodes distance k from a leaf
        self.nodes_dist_k_from_leaf = set()
        for leaf in self.leaves:
            # since there are no nodes reachable from a leaf, take the leaf's predecessor and make the cutoff k-1
            for node in graph.predecessors(leaf):
                paths = nx.single_source_shortest_path_length(graph, node, cutoff=k-1)
                for key in paths:
                    self.nodes_dist_k_from_leaf.add(key)

        self.nnodes_dist_k_from_leaf = len(self.nodes_dist_k_from_leaf)

    def to_vector(self):
        '''

        :return: Converted graph features to a vector, containing the following
        entries:

            nroots                  1
            nleaves                 1
            root_type_counts        len(Entity.TYPES)
            leaf_type_counts        len(Entity.TYPES)
            node_type_counts        len(Entity.TYPES)
            edge_type_counts        len(Relations.TYPES)
            all_paths_type_counts   len(paths) x len(Entity.TYPES)
            avg_len_path            1
            max_len_path            1
            avg_children_counts     1
            max_children_counts     1
            entity_relations_mat    len(Entity.TYPES) x len(Entity.TYPES) x
                                    len(Relations.TYPES)
            nnodes_dist_k_from_root 1
            nnodes_dist_k_from_leaf 1

        '''
        vec = []
        vec.append(self.nroots)
        vec.append(self.nleaves)
        vec.extend([val for val in self.root_type_counts.values()])
        vec.extend([val for val in self.leaf_type_counts.values()])
        vec.extend([val for val in self.node_type_counts.values()])
        vec.extend([val for val in self.edge_type_counts.values()])
        vec.extend([val for val in self.all_paths_type_counts.values()])
        vec.append(self.avg_len_path)
        vec.append(self.max_len_path)
        vec.append(self.avg_children_counts)
        vec.append(self.max_children_counts)
        vec.extend(self.entity_relations_mat.flatten())
        vec.append(self.nnodes_dist_k_from_root)
        vec.append(self.nnodes_dist_k_from_leaf)
        return np.array(vec)


def aggregate_graph_features(k):
    """

    :return: Aggregate GraphFeatures, constructed for all graphs
    stored at '../UKP/graphs', in a dictionary of the following form:

        {
            <str essay_name> : <GraphFeature object>
            ...
        }

    """
    from parser import Entity, Attribute, Relations
    os.chdir('../UKP/graphs')
    graph_features_agg = {}
    feature_vectors_agg = {}
    iter = 0
    for file in os.listdir():
        if file.endswith("gpickle"):
            G = nx.read_gpickle(file)
            fname = file.split('.')[0]
            graph_features_agg[fname] = GraphFeatures(G,k)
            feature_vectors_agg[fname] = graph_features_agg[fname].to_vector()
            pdb.set_trace()
            print("done with iter {}".format(iter))
            iter += 1

    os.chdir('../../Team_BART')
    with open('graph_features_agg', 'wb') as f:
        pickle.dump(graph_features_agg,f)
    with open('feature_vectors_agg', 'wb') as f:
        pickle.dump(feature_vectors_agg,f)

def plot_stacked_box_hist(graph_features_agg, data, suptitle, vline_label, xlabel, ylabel):
    sns.set(style="ticks")
    fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
    plt.suptitle(suptitle)

    sns.boxplot(data, ax=ax_box, color='lightsteelblue')
    # sns.distplot(avg_path_lens, ax=ax_hist, bins = np.arange(np.amax(avg_path_lens), step = 50), norm_hist = False)
    plt.hist(data, color='lightsteelblue')
    ax_hist.axvline(x = np.mean(data), linestyle = "--", color = "darkblue", label = vline_label)
    ax_box.set(yticks=[])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    legend = ax_hist.legend(loc='upper right')
    sns.despine(ax=ax_hist)
    sns.despine(ax=ax_box, left=True)

    plt.show()

if __name__ == "__main__":
    aggregate_graph_features(10)

    loaded_agg_features = pickle.load( open( "graph_features_agg", "rb" ) )

    # avg_path_lens = [loaded_agg_features[gf].avg_len_path for gf in loaded_agg_features.keys()]
    # plot_stacked_box_hist(loaded_agg_features, avg_path_lens, "Mean Root-to-Leaf Path Length ({} Graphs)".format(len(loaded_agg_features)), "Mean Length: {}".format(np.mean(avg_path_lens)), "Length", "Length Frequency")

    # max_path_lens = [loaded_agg_features[gf].max_len_path for gf in loaded_agg_features.keys()]
    # plot_stacked_box_hist(loaded_agg_features, max_path_lens, "Max Root-to-Leaf Path Length ({} Graphs)".format(len(loaded_agg_features)), "Mean Maximum Length: {}".format(np.mean(max_path_lens)), "Length", "Length Frequency")

    nroots = [loaded_agg_features[gf].nroots for gf in loaded_agg_features.keys()]
    plot_stacked_box_hist(loaded_agg_features, nroots, "Root Node Counts ({} Graphs)".format(len(loaded_agg_features)), "Mean Root Count: {}".format(np.mean(nroots)), "Count", "Count Frequency")
