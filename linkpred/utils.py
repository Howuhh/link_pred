import sys

import numpy as np
from tqdm import tqdm
sys.path.append("/usr/local/lib/python3.7/site-packages")
import graph_tool as gt


def feature_vector(G, scores, delim=","):
    """
    Converts node feature_vector from list of scores and add target label. 
    A label is 1 if the edge between node pair exists and 0 otherwise.

    Parameters
    ----------
    G: graph_tool.Graph
        Graph object from graph-tool module.
    scores: ndarray
        Metric scores for patricular node pair.
    delim: str
        Delimiter in joined string.

    Returns
    -------
    feature_vector: str
        A string representing node scores + target label.

    """
    target = 0.
    if G.edge(scores[0], scores[1]):
        target = 1.

    vector = f"{delim}".join(str(i) for i in scores)
    label = f"{delim}{target}"

    return vector + label


def compute_metrics(g_neighbors, metrics, node_pair):
    """
    Computes all similarity scores for node pair.

    Parameters
    ----------
    g_neighbors: dict
        Dictionary in format {node_id: array_of_neighbors},
        where value should be ndarray type.
    metrics: list
        List with metrics funcitons.
        For the right function format see docs in metrics.py
    node_pair: tuple
        Pair of nodes, for which the metrics of similarity are calculated.

    Returns
    -------
    scores: ndarray, shape (1, len(metrics))
        All similarity scores for node pair.

    """
    node1, node2 = node_pair
    result = np.zeros(len(metrics), dtype=np.float)

    for i, metric in enumerate(metrics):
        result[i] = np.round(metric(g_neighbors, node1, node2), 3)

    return result


def graph_neighbors(G, k_neighbors=2):
    """
    Computes all neighbors for each node and represents it as a dictionary.

    Parameters
    ----------
    G: graph_tool.Graph
        Graph object from graph-tool module.
    k_neighbors=2: int
        K-th order neighbors for each node. Only 1 or 2.

    Returns
    -------
    nodes_neighbors: dict
        dictionary with node id as key and ndarray of neighbors ids.

    """
    assert k_neighbors in (1, 2), "only 1 and 2 neighbors"

    nodes_neighbors = {}

    for node in tqdm(G.vertices()):
        neighbors_ = neighbors(G, node, k_neighbors)
        nodes_neighbors[node] = neighbors_

    return nodes_neighbors


def neighbors(G, node, k_neighbors):
    """
    Returns all node neighbors.

    Parameters
    ----------
    G: graph_tool.Graph
        Graph object from graph-tool module.
    node: int
        node id in graph
    k_neighbors: int
        K-th order neighbors. Only 1 or 2.

    Returns
    -------
    neighbors: ndarray
        Node neighbors.

    """
    initn = list(G.get_out_neighbors(node))
    visited = []

    if k_neighbors == 1:
        return G.get_out_neighbors(node)

    for node_ in initn:
        visited.extend(G.get_out_neighbors(node_))

    true_neighborhood = np.setdiff1d(visited, initn + [node])

    return true_neighborhood


# well, this is also weird, 
# but faster than by hands from edgelsit
def graph_edges_split(G, p):
    """
    Split graph edges for validation and training disjoint sets.

    Parameters
    ----------
    G: graph_tool.Graph
        Graph object from graph-tool module.
    p: int, (0, 1]
        Test proportion.
    Returns
    -------
    train_graph, test_graph: graph_tool.Graph
    
    """
    N = G.num_edges()
    K = np.int(N * p)

    train_mask = np.array([0] * K + [1] * (N-K), dtype=np.bool)
    np.random.shuffle(train_mask)
    test_mask = ~train_mask

    train_graph = gt.GraphView(G, directed=False)
    test_graph = gt.GraphView(G, directed=False)

    prop_train = train_graph.new_edge_property("bool")
    prop_train.a = train_mask

    prop_test = test_graph.new_edge_property("bool")
    prop_test.a = test_mask

    train_graph.set_edge_filter(prop_train)
    test_graph.set_edge_filter(prop_test)

    return train_graph, test_graph