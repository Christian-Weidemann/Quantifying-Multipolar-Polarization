import sys
import numpy as np
import networkx as nx
from collections import defaultdict

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

# Function creating the o vector
def make_o(size, factor):
   o = np.random.normal(size = size, loc = factor, scale = 0.2)
   o[o > 1] = 1 - (o[o > 1] - 1)
   o = np.concatenate([o, -o])
   o.sort()
   return {i: o[i] for i in range(o.shape[0])}

# List of tested p_outs and the corresponding p_ins that will generate an SBM with roughly the same number of edges
out_p = (0.0085, 0.008, 0.0075, 0.007, 0.0065, 0.006,  0.0055, 0.005,   0.0045, 0.004,  0.0035, 0.003,   0.0025, 0.002,  0.0015, 0.001,   0.0005, 0.0001)
in_p =  (0.0085, 0.012, 0.0155, 0.019, 0.0226, 0.0261, 0.0296, 0.03316, 0.0367, 0.0402, 0.0437, 0.04726, 0.0508, 0.0543, 0.0578, 0.06136, 0.0648, 0.0678)

deltas = defaultdict(list)
for run in range(10):                                                     # We do 10 independent runs...
   for _ in range(len(out_p)):                                            # ... for each p_out value...
      sys.stderr.write(f"{run} {out_p[_]}\n")
      probs = np.full((8, 8), out_p[_])                                   # Make the SBM and its o vector
      np.fill_diagonal(probs, in_p[_])
      G = nx.stochastic_block_model(sizes = [125] * 8, p = probs)
      while nx.number_connected_components(G) > 1:
         G = nx.stochastic_block_model(sizes = [125] * 8, p = probs)
      o = make_o(len(G.nodes) // 2, 0.8)
      nx.set_node_attributes(G, o, "polar")
      deltas[_].append(nx.numeric_assortativity_coefficient(G, "polar"))  # Calculate assortativity
      nx.set_edge_attributes(G, {e: (G.nodes[e[0]]["polar"] + G.nodes[e[1]]["polar"]) / 2 for e in G.edges}, "polar")

with open("assort_sensitivity.csv", 'w') as f:
   for _ in range(len(out_p)):
      f.write(f"{out_p[_]}\t{np.mean(deltas[_])}\t{np.std(deltas[_])}\n")
