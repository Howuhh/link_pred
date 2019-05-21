import sys

import numpy as np
from tqdm import tqdm

from .utils import graph_edges_split, graph_neighbors
from .utils import feature_vector, compute_metrics

sys.path.append("/usr/local/lib/python3.7/site-packages")
import graph_tool.all as gtl


def extract_features(G, metrics, file, p=0.1, k_neighbors=2):
    """
    Extract all features/metrics from a graph and writes to the CSV file
    For each node similarity metrics are computed only for candidates on distance 2(3) from root node.
    For each node only top 40 most similar nodes are left. 
    File format: (from_node, to_node, metrics, label)

    Parameters
    ----------
    G: graph_tool.Graph
         Graph object from graph-tool module. See https://graph-tool.skewed.de/.
    metrics: list
        List with node similarity funcitons.
    file: str
        Path & Name for final csv with features.
    p=0.1: int, (0, 1]
        Propotion of deleted edges for prediction.
    k_neighbors=2: int
        K-th order neighbors for each node. Only 1 or 2.

    Returns
    -------
    None, writes data to a file.

    """
    train_graph, test_graph = graph_edges_split(G, p=p)
    nodes_info = graph_neighbors(train_graph, k_neighbors=k_neighbors)

    with open(file, "w") as df:
        for i, node in tqdm(enumerate(train_graph.get_vertices())):
            scores = _get_node_features(train_graph, nodes_info, metrics, node)

            for row in scores:
                row_str = feature_vector(test_graph, row)
                df.write(row_str + "\n")
    print("DONE")


def _get_node_features(G, g_neighbors, metrics, node):
    """
    Helper funcion.
    Computes similatiry for all node pairs (root_node, node_i) within distance 2(or 3) from root node.
    Only top 40 (by row sum) pairs are left.

    Parameters
    ----------
    G: graph_tool.Graph
        Graph object from graph-tool module. See https://graph-tool.skewed.de/.
    g_neighbors: dict
        Dictionary in format {node_id: array_of_neighbors}
    metrics: list
        List with node similarity funcitons.
    node: int
        Node id for which features the metrics are calculated. 

    Returns
    -------
    top_scores: ndarray, shape (40, len(metrics) + 2)
        Row format: [from_node, to_node, *metrics]

    """
    # get all nodes within distance 2 from root node
    candidates = gtl.shortest_distance(G, node, max_dist=2, return_reached=True)[1]
    result = np.zeros((candidates.shape[0], len(metrics) + 2))

    for i, cand in enumerate(candidates):
        if cand == node:
            continue
        scores = compute_metrics(g_neighbors, metrics, (node, cand))
        result[i, :2] = [node, cand]
        result[i, 2:] = scores

    # that's weird! sorting accoring row sum
    top_n = result[result[:, 2:].sum(axis=1).argsort()][-40:]

    return top_n
