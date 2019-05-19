import numpy as np


def common_neighbors_score(g_neighbors, node1, node2):
    """
    Set docstring here.

    Parameters
    ----------
    g_neighbors: dict
    node1: int
    node2: int

    Returns
    -------

    """
    common_n = _common_neighbors(g_neighbors, node1, node2)

    return common_n.shape[0]


def _common_neighbors(g_neighbors, node1, node2):
    """
    Set docstring here.

    Parameters
    ----------
    g_neighbors: dict
    node1: int
    node2: int

    Returns
    -------

    """
    node1_n = g_neighbors[node1]
    node2_n = g_neighbors[node2]

    common_n = np.intersect1d(node1_n, node2_n, assume_unique=True)

    return common_n


def adamic_adar_score(g_neighbors, node1, node2):
    """
    Set docstring here.

    Parameters
    ----------
    g_neighbors: dict
    node1: int
    node2: int

    Returns
    -------

    """
    common_n = _common_neighbors(g_neighbors, node1, node2)
    degrees = _common_degree(g_neighbors, common_n)

    # otherwise gives too much weight to second friends without friends
    inv_log = np.divide(1., np.log(degrees + 1e-2))
    inv_log[inv_log < 0] = 0

    return np.sum(inv_log)


def _common_degree(g_neighbors, common):
    """
    Set docstring here.

    Parameters
    ----------
    g_neighbors: dict
    common: int

    Returns
    -------

    """
    N = common.shape[0]
    degrees = np.zeros(N, dtype=np.int)

    degrees[:] = [g_neighbors[node].shape[0] for node in common]

    return degrees


def res_allocation(g_neighbors, node1, node2):
    """
    Set docstring here.

    Parameters
    ----------
    g_neighbors: dict
    node1: int
    node2: int

    Returns
    -------

    """
    common_n = _common_neighbors(g_neighbors, node1, node2)
    degrees = _common_degree(g_neighbors, common_n)

    score = np.sum(np.divide(1., degrees + 1e-2))

    return score
