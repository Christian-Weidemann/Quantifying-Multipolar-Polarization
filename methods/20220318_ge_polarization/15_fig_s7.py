import random, sys
import numpy as np
import pandas as pd
import networkx as nx
from modules import ps

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

# This function makes the o vector given the size of the community (comm_size), the number of communities (comm_n), the value of mu (factor) and the level of purity (purity).
# If rnd_G is True then the underlying graph for this o is random, and thus there is no need to perform some of the operations.
def make_o(comm_size, comm_n, factor, purity, rnd_G = False):
   o = np.random.normal(size = comm_size * comm_n, loc = factor, scale = 0.2) # Make a random opinion distribution with mean "factor" and std = 0.2.
   o[o > 1] = 1 - (o[o > 1] - 1)                                              # Make sure no entry is larger than 1
   o = np.concatenate([o, -o])                                                # Symmetrize the opinion vector to make the negative block
   if not rnd_G:
      o.sort()                                                                # Sort o by value
      np.random.shuffle(o[:(comm_size * comm_n)])                             # Shuffle the o values on the left side of the opinion vector
      np.random.shuffle(o[(comm_size * comm_n):])                             # Shuffle the o values on the right side of the opinion vector
      if purity < 1:                                                          # If purity = 1 no action is necessary: communities will get o values exclusively from the same sign
         diversity = 1 - purity                                               # If purity < 1, we need to swap o values with opposite signs to create impure communities
         c_pair = random.sample(list(range(comm_n)), comm_n)                  # Get random community pairings so we swap between communities at an unpredictable distance from the middle
         swapping_indexes = np.array(random.sample(list(range(comm_size)), round(diversity * (comm_size / 2)))) # IDs of the nodes inside a community that we'll swap, dependent on diversity
         for c in range(comm_n):
            myn = (c * comm_size) + swapping_indexes                          # Find the indexes in o corresponding to the node IDs in the community I need to swap
            oppn = ((c_pair[c] + 4) * comm_size) + swapping_indexes           # Same thing but for the community I'm swapping with
            o[myn], o[oppn] = o[oppn], o[myn]                                 # Swap the o values
   return {i: o[i] for i in range(o.shape[0])}

comm_n = 8
comm_size = 128
df = []
for run in range(50):
   sys.stderr.write(f"{run}\n")
   row = []
   ### Make a network with strong "comm_n" communities with "comm_size" nodes, making sure it's a single connected component
   probs = np.full((8, 8), 0.0003)
   np.fill_diagonal(probs, 0.067)
   G = nx.stochastic_block_model(sizes = [comm_size] * comm_n, p = probs)
   while nx.number_connected_components(G) > 1:
      G = nx.stochastic_block_model(sizes = [comm_size] * comm_n, p = probs)
   ###
   rnd = G.copy()
   rnd = nx.double_edge_swap(rnd, nswap = len(rnd.edges) * 8, max_tries = len(rnd.edges) * 800) # Make a randomized version of rthe network without communities, by making lots of edge swaps
   while nx.number_connected_components(rnd) > 1:
      rnd = nx.double_edge_swap(rnd)
   Q = ps._ge_Q(G)
   row.append(ps.ge(make_o(comm_size, comm_n // 2, 0.8, None, rnd_G = True), {}, rnd))          # Calculate the polarization of the random networks
   for purity in (0.0, 0.25, 0.5, 0.75, 1.0):                                                   # For different levels of purity...
      row.append(ps.ge(make_o(comm_size, comm_n // 2, 0.8, purity), {}, G, Q = Q))              # ... calculate polarization
   df.append(row)

df = pd.DataFrame(data = df, columns = ("rnd", 0.0, 0.25, 0.5, 0.75, 1.0))

df2 = pd.DataFrame()
df2["mean"] = df.mean()
df2["std"] = df.std()

df2.reset_index().reset_index().to_csv("purity.csv", index = False, sep = "\t")
