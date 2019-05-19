import sys

import numpy as np
from tqdm import tqdm

from .utils import graph_edges_split, graph_neighbors
from .utils import feature_vector, compute_metrics

sys.path.append("/usr/local/lib/python3.7/site-packages")
import graph_tool.all as gtl


def extract_features(G, metrics, file, p=0.1, k_neighbors=2):
    train_graph, test_graph = graph_edges_split(G, p=p)
    nodes_info = graph_neighbors(train_graph, k_neighbors=k_neighbors)

    with open(file, "w") as df:
        for i, node in tqdm(enumerate(train_graph.get_vertices())):
            scores = get_node_features(train_graph, nodes_info, metrics, node)

            for row in scores:
                row_str = feature_vector(test_graph, row)
                df.write(row_str + "\n")
    print("DONE")


def get_node_features(G, g_neighbors, metrics, node):
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
