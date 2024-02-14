import random, sys
import numpy as np
import networkx as nx
from modules import ps

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

# This function estimates the polarization of a network with a given number of nodes (nodes), communities (comms), and p_out value.
def polarization_at(nodes, comms, p_out):
   csize = nodes // comms                                   # Determine the size of each community
   G1 = nx.complete_graph(csize)                            # Make the community as a clique
   G = G1.copy()
   for _ in range(comms - 1):                               # Join together "comms" cliques
      G = nx.disjoint_union(G, G1.copy())
   o = {n: -1 if n < len(G.nodes) / 2 else 1 for n in G}    # Nodes in half of the communities get an o value of -1 and the other half communities gets a +1
   n_between_edges = round(len(G.edges) * p_out)            # Determine how many edges must be between communities given p_out
   old_edge_count = len(G.edges)
   while (len(G.edges) - old_edge_count) < n_between_edges: # Keep going until we added the required number of edges
      c1, c2 = random.sample(list(range(comms)), 2)         # Pick two random communities
      n1 = (c1 * csize) + random.randint(0, csize - 1)      # Pick a random node in one community
      n2 = (c2 * csize) + random.randint(0, csize - 1)      # Pick a random node in the other community
      G.add_edge(n1, n2)                                    # Connect the two nodes
   return ps.ge(o, {}, G)

# params: # nodes, # communities, p_out
with open("fragmentation.csv", 'w') as f:
   f.write("comms\tpol_mean\tpol_std\n")
   for comms in range(16, 1, -2):                                     # Repeat the experiment for 16, 14, 12, 10, 8, 6, 4, 2 communities
      sys.stderr.write(f"{comms}\n")
      pols = [polarization_at(1680, comms, 0.05) for _ in range(10)]
      f.write(f"{comms}\t{np.mean(pols)}\t{np.std(pols)}\n")

