import sys

import numpy as np
from tqdm import tqdm
sys.path.append("/usr/local/lib/python3.7/site-packages")
import graph_tool as gt


def feature_vector(G, scores, delim=","):
    """
    Set docstring here.

    Parameters
    ----------
    G: graph_tool.Graph
    scores: ndarray
    delim: str

    Returns
    -------

    """
    target = 0.
    if G.edge(scores[0], scores[1]):
        target = 1.

    vector = f"{delim}".join(str(i) for i in scores)
    label = f"{delim}{target}"

    return vector + label


def compute_metrics(g_neighbors, metrics, node_pair):
    """
    Set docstring here.

    Parameters
    ----------
    g_neighbors: dict
    metrics: list
    node_pair: tuple

    Returns
    -------

    """
    node1, node2 = node_pair
    result = np.zeros(len(metrics), dtype=np.float)

    for i, metric in enumerate(metrics):
        result[i] = np.round(metric(g_neighbors, node1, node2), 3)

    return result


def graph_neighbors(G, k_neighbors=2):
    """
    Set docstring here.

    Parameters
    ----------
    G: graph_tool.Graph
    k_neighbors: int 

    Returns
    -------

    """
    assert k_neighbors in (1, 2), "only 1 and 2 neighbors"

    nodes_neighbors = {}

    for node in tqdm(G.vertices()):
        neighbors_ = neighbors(G, node, k_neighbors)
        nodes_neighbors[node] = neighbors_

    return nodes_neighbors


def neighbors(G, node, k_neighbors):
    """
    Set docstring here.

    Parameters
    ----------
    G: graph_tool.Graph
    node: int
    k_neighbors: int

    Returns
    -------

    """
    initn = list(G.get_out_neighbors(node))
    visited = []

    if k_neighbors == 1:
        return G.get_out_neighbors(node)

    for node_ in initn:
        visited.extend(G.get_out_neighbors(node_))

    true_neighborhood = np.setdiff1d(visited, initn + [node])

    return true_neighborhood


# well, this is also weird
def graph_edges_split(G, p):
    """
    Set docstring here.

    Parameters
    ----------
    G: graph_tool.Graph
    p: int, must be in range (0, 1]

    Returns
    -------

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