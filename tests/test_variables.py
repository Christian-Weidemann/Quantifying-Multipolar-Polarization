"""
Variables which are common to all synthetic experiments.
"""
import networkx as nx
import tests.methods.utils as utils
import matplotlib.pyplot as plt
import tests.methods.torch_nvd as nvd
import random

random.seed(0)

metrics = [('APD', nvd.pairwise_average), 
           ('ADM', nvd.avg_dist_to_mean), 
           ('PC', nvd.PCA_manifold), 
           ('MDS_eu', nvd.MDS_euclidean_manifold),
           ('MDS_er', nvd.MDS_er_manifold),
           ('TV', nvd.total_variation)]

local_tests = [('Chains', nx.path_graph), 
                ('Complete graphs', nx.complete_graph),
                ('Communities', utils.make_community_graph)]

local_num_nodes_range = range(6, 13, 2)

random_run_metrics = [nvd.MDS_euclidean_manifold, nvd.MDS_er_manifold]

# Number of MDS runs with random initializations
mds_runs = 50

# Number of random SBM graph initializations
num_runs = 20

# Set look of networks
node_size = 50
edge_color = 'grey'
local_edge_width = 1.5
local_edge_alpha = 0.5
global_edge_width = 1
global_edge_alpha = 0.5

# set matplotlib font to Arial with medium weight
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.weight'] = 'medium'