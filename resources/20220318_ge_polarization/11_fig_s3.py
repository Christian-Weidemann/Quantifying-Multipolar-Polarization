import random
import numpy as np
import networkx as nx
from modules import ps

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

def polarization_at(n):
   G1 = nx.complete_graph(n)                               # Create a clique of size n
   G2 = G1.copy()                                          # Make a second clique of size n
   G = nx.disjoint_union(G1, G2)                           # Create a graph made of the two (isolated) cliques we just creates
   o = {n: -1 if n < len(G.nodes) / 2 else 1 for n in G}   # Give -1 o values to all nodes in one clique and +1 to all nodes of the other clique
   n_between_edges = round(len(G.edges) * 0.05)            # Decide how many edges to establish between cliques (5% of all edges in the two cliques)
   old_edge_count = len(G.edges)
   while (len(G.edges) - old_edge_count) < n_between_edges:# Pick the appropriate number of edges to create by randomly extracting nodes ids from the two cliques
      G.add_edge(random.randint(0, len(G1.nodes) - 1), random.randint(len(G1.nodes), (len(G1.nodes) * 2) - 1))
   return ps.ge(o, {}, G)                                  # Calculate the network's delta polarization

# For each n value (clique size), repeat the experiment ten times and take the average polarization score, to smooth out randomness
deltas = {n: np.mean([polarization_at(n) for _ in range(10)]) for n in range(5, 100)}

# Write output to file
with open("block_delta.csv", 'w') as f:
   f.write("blocksize\tdelta\n")
   for n in range(5, 100):
       f.write(f"{n}\t{deltas[n]}\n")

def polarization_at2(n, k):
   G1 = nx.complete_graph(n)                               # Create a clique of size n
   G1.remove_edges_from(random.choices(list(G.edges), k = k))
   G2 = G1.copy()                                          # Make a second clique of size n
   G = nx.disjoint_union(G1, G2)                           # Create a graph made of the two (isolated) cliques we just creates
   o = {n: -1 if n < len(G.nodes) / 2 else 1 for n in G}   # Give -1 o values to all nodes in one clique and +1 to all nodes of the other clique
   n_between_edges = round(len(G.edges) * 0.05)            # Decide how many edges to establish between cliques (5% of all edges in the two cliques)
   old_edge_count = len(G.edges)
   while (len(G.edges) - old_edge_count) < n_between_edges:# Pick the appropriate number of edges to create by randomly extracting nodes ids from the two cliques
      G.add_edge(random.randint(0, len(G1.nodes) - 1), random.randint(len(G1.nodes), (len(G1.nodes) * 2) - 1))
   return ps.ge(o, {}, G)     
