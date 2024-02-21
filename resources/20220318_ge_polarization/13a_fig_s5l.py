import random, sys
import numpy as np
import networkx as nx
from modules import ps

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

G1 = nx.complete_graph(840)                              # Make a clique
G = G1.copy()                                            # Copy the clique
G = nx.disjoint_union(G, G1.copy())                      # Make a network with the two cliques
o = {n: -1 if n < len(G.nodes) / 2 else 1 for n in G}    # Nodes in one clique get a o value of -1, nodes in the other get a value of +1
n_between_edges = round(len(G.edges) * 0.05)             # Figure out how many edges there should be between cliques to have p_out = 0.05
old_edge_count = len(G.edges)
while (len(G.edges) - old_edge_count) < n_between_edges: # Cycle until we added all the required edges
   n1 = random.randint(0, 840 - 1)                       # Pick a random node from the first clique
   n2 = 840 + random.randint(0, 840 - 1)                 # Pick a random node from the other clique
   G.add_edge(n1, n2)                                    # Add an edge between the two

edge_count = len(G.edges)
with open(f"density{sys.argv[1]}.csv", 'w') as f:        # Should pass a run ID to the script so that you can run in parallel and then average the results afterwards
   Gop = G.copy()
   f.write("removed\tpol\n")
   f.write(f"0.0\t{ps.ge(o, {}, Gop)}\n")                # Calculate the pol without removing edges
   for frac in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.5, 0.75, 0.9):
      sys.stderr.write(f"{frac}\n")
      Gop = G.copy()
      Gop.remove_edges_from(random.sample(list(G.edges), np.ceil(len(G.edges) * frac).astype(int))) # Remove "frac" edges from the network
      f.write(f"{1 - (len(Gop.edges) / edge_count)}\t{ps.ge(o, {}, Gop)}\n")                        # Calculate polarization after removing "frac" edges

