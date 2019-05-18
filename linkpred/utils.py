import pickle

import numpy as np
import networkx as nx

from tqdm import tqdm


def neighbors(G, node):
    initn = list(G.neighbors(node))
    visited = []
    
    for node_ in initn:
        visited.extend(G.neighbors(node_))
    
    true_neighborhood = np.setdiff1d(visited, initn + [node])
        
    return true_neighborhood

def graph_neighbors(G):
    nodes_neighbors = {}
    
    for node in tqdm(G.nodes):
        neighbors_ = neighbors(G, node)
        nodes_neighbors[node] = np.array(neighbors_, dtype=np.int)
    
    return nodes_neighbors


def dump_file(path, obj):
    with open(path, "wb") as file:
        pickle.dump(obj, file)
        

def load_file(path): 
    with open(path, "rb") as file:
        data = pickle.load(file)
        
    return data




# def neighbors(G, node):
#     initn = list(G.neighbors(node))
#     visited = set()
    
#     for node_ in initn:
#         visited.update(G.neighbors(node_))
#     true_neighborhood = visited.difference(initn + [node])
        
#     return true_neighborhood