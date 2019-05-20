import numpy as np


def common_neighbors_score(g_neighbors, node1, node2):
    """
    Similarity measure based on number of common neighbors between two nodes.

    Parameters
    ----------
    g_neighbors: dict
        Dictionary in format {node_id: array_of_neighbors},
    node1: int
        Node id
    node2: int
        Node id

    Returns
    -------
    score: int
        Nodes similarity score.

    """
    common_n = _common_neighbors(g_neighbors, node1, node2)

    return common_n.shape[0]


def _common_neighbors(g_neighbors, node1, node2):
    """
    Helper function. Computes common neighbors.

    Parameters
    ----------
    g_neighbors: dict
        Dictionary in format {node_id: array_of_neighbors},
    node1: int
        Node id
    node2: int
        Node id

    Returns
    -------
    common: ndarray
        Array of common neighbors ids for node1 and node2.

    """
    node1_n = g_neighbors[node1]
    node2_n = g_neighbors[node2]

    common_n = np.intersect1d(node1_n, node2_n, assume_unique=True)

    return common_n


def adamic_adar_score(g_neighbors, node1, node2):
    """
    Similarity measure. Defined as the sum of the inverse logarithmic
    degree centrality of the neighbours shared by the two nodes.
    Introduced in 2003 by Lada Adamic and Eytan Adar.

    Parameters
    ----------
    g_neighbors: dict
        Dictionary in format {node_id: array_of_neighbors},
    node1: int
        Node id
    node2: int
        Node id

    Returns
    -------
    score: int
        Nodes similarity score.

    """
    common_n = _common_neighbors(g_neighbors, node1, node2)
    degrees = _common_degree(g_neighbors, common_n)

    # otherwise gives too much weight to second friends without friends
    inv_log = np.divide(1., np.log(degrees + 1e-2))
    inv_log[inv_log < 0] = 0

    return np.sum(inv_log)


def _common_degree(g_neighbors, common):
    """
    Helper functoin. Computes degrees of common neighbors.

    Parameters
    ----------
    g_neighbors: dict
    common: int

    Returns
    -------
    degrees: ndarray
        Array of node degrees.

    """
    N = common.shape[0]
    degrees = np.zeros(N, dtype=np.int)

    degrees[:] = [g_neighbors[node].shape[0] for node in common]

    return degrees


def res_allocation(g_neighbors, node1, node2):
    """
    Similarity measure based on resource allocation.
    For a detailed explanation see Tao Zhou, 
    Linyuan Lu ̈ and Yi-Cheng Zhang paper: https://arxiv.org/abs/0901.0553

    Parameters
    ----------
    g_neighbors: dict
        Dictionary in format {node_id: array_of_neighbors},
    node1: int
        Node id
    node2: int
        Node id

    Returns
    -------
    score: int
        Nodes similarity score.

    """
    common_n = _common_neighbors(g_neighbors, node1, node2)
    degrees = _common_degree(g_neighbors, common_n)

    score = np.sum(np.divide(1., degrees + 1e-2))

    return score
