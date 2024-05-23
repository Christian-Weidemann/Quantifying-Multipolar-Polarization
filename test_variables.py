"""
Variables which are common to all synthetic experiments.
"""
import networkx as nx
import methods.utils as utils
import matplotlib.pyplot as plt
import methods.torch_nvd as nvd
import random

# Seed for reproducibility of visualizations / random graph generation
random.seed(0)

metrics = [('APD', nvd.pairwise_average), 
           ('ADM', nvd.avg_dist_to_mean), 
           ('PC', nvd.PC_manifold), 
           ('MDS', nvd.MDS_manifold),
           ('TV', nvd.total_variation)]

local_tests = [('Chains', nx.path_graph), 
                ('Complete', nx.complete_graph),
                ('Communities', utils.make_community_graph)]

local_num_nodes_range = range(6, 13, 2)

random_run_metrics = [nvd.MDS_manifold]

# Number of MDS runs with random initializations
mds_runs = 100

# Number of random SBM graph initializations
SBM_runs = 10

# Set look of networks
node_size = 50
edge_color = 'grey'
local_edge_width = 1.5
local_edge_alpha = 0.5
global_edge_width = 1
global_edge_alpha = 0.5

# matplotlib font and text settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.weight'] = 'medium'

labelpad = 20
fontsize = 12