import numpy as np


def common_neighbors_score(g_neighbors, node1, node2):
    common_n = _common_neighbors(g_neighbors, node1, node2)

    return common_n.shape[0]


def _common_neighbors(g_neighbors, node1, node2):
    node1_n = g_neighbors[node1]
    node2_n = g_neighbors[node2]

    common_n = np.intersect1d(node1_n, node2_n, assume_unique=True)

    return common_n


def adamic_adar_score(g_neighbors, node1, node2):
    common_n = _common_neighbors(g_neighbors, node1, node2)
    degrees = _common_degree(g_neighbors, common_n)

    inv_log = np.divide(1., np.log(degrees))

    return np.sum(inv_log)


def _common_degree(g_neighbors, common):
    N = common.shape[0]
    degrees = np.zeros(N, dtype=np.int)

    degrees[:] = [g_neighbors[node].shape[0] for node in common]

    return degrees


def res_allocation(g_neighbors, node1, node2):
    common_n = _common_neighbors(g_neighbors, node1, node2)
    degrees = _common_degree(g_neighbors, common_n)

    score = np.sum(np.divide(1., degrees))

    return score
