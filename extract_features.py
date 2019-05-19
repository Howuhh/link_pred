import sys
import numpy as np

from linkpred.metrics import common_neighbors_score, adamic_adar_score
from linkpred.metrics import res_allocation
from linkpred.features import extract_features

sys.path.append("/usr/local/lib/python3.7/site-packages")
import graph_tool as gt

METRICS = [
    common_neighbors_score,
    adamic_adar_score,
    res_allocation,
]

if __name__ == "__main__":
    edge_list = np.loadtxt("data/facebook_combined.txt", delimiter=" ")
    gt_fc = gt.Graph()
    gt_fc.add_edge_list(edge_list)

    extract_features(gt_fc, METRICS, "data/fc_features_first.csv", k_neighbors=1)
