import sys
import numpy as np
import pandas as pd
import networkx as nx
from modules import ps
from collections import defaultdict

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

# This function creates a multipolar opinion vector o. It requires three inputs:
# - size: half the number of the nodes in the network;
# - factor: the absolute value of the average polarity of each side (mu in the paper)
# - ncomms: the number of different poles
# If factor = 0 there is no opinion polarization, with factor = 1 opinions cluster at +/- 1
def make_os(size, factor, ncomms):
   o = np.random.normal(size = size, loc = factor, scale = 0.2) # Make a random normal distribution with std = 0.2
   o[o > 1] = 1 - (o[o > 1] - 1)                                # Mirror values out of the +/-1 bounds to be inbound
   o[o < 0] *= -1                                               # Make sure all values are positive
   os = {}                                                      # Make n o vectros, one per pole, which are copies of o for the nodes in a specific community and 0 otherwise
   for i in range(ncomms):
      for j in range(size):
         os[j + (i * size)] = np.zeros(ncomms)
         os[j + (i * size)][i] = o[j]
   return os

# This function modifies the community connection probability in the SBM
# Specifically, it sets to zero the probabilities between communities too far
# in the polarity specturm. Takes as input:
# - p: the original SBM connection probability (a CxC matrix)
# - k: how many communities a community connects to. This is n in the main paper.
def update_p(p, k):
   p_sum = p.sum()                                                                           # Save the sum of p entries, this must be constant to ensure same expected # of edges
   for col in range(8):                                                                      # For every column...
      for row in range(8):                                                                   # ...and every row...
         if ((col < 8 - k) and ((row - col) > k)) or ((row < 8 - k) and ((col - row) > k)):  # ...find the entries that are k-1 steps away from the diagonal...
            p[row, col] = 0                                                                  # ...and set them to zero
   p *= (p_sum / p.sum())                                                                    # Make sure that the new p sums to the same value as the old one
   return p

run = int(sys.argv[1])  # This can be used to run mutliple independently initialized runs

results = []                                             # Stores all numerical results per parameter combination: delta, assortativity, and RWC
factors = (0.0, 0.2, 0.4, 0.6, 0.8)                      # All testes mu values
in_p =  (0.0085,  0.039,  0.054,  0.062,  0.064,  0.067) # All tested p_in values
out_p = (0.0085, 0.0042, 0.0024, 0.0012, 0.0006, 0.0003) # All tested p_out values

for i in range(len(in_p)):                                                       # Loop over all possible p_in-p_out pairs
   probs = np.full((8, 8), out_p[i])                                             # Initialize SBM connection probabilities with p_out
   np.fill_diagonal(probs, in_p[i])                                              # Set diagonal elements of connection probability matrix to be equal to p_in
   for conn_neigh in range(7, 0, -1):                                            # Loop from n = 7 to n = 1
      probs = update_p(probs, conn_neigh)                                        # Change the matrix of community connection probability p by removing connections between disagreeing communities
      G = nx.stochastic_block_model(sizes = [125] * 8, p = probs)                # Initialize SBM
      while nx.number_connected_components(G) > 1:                               # If the SBM has more than one connected component...
         G = nx.stochastic_block_model(sizes = [125] * 8, p = probs)             # ...reinitialize it, otherwise we cannot compute delta or RWC.
      Q = ps._ge_Q(G)                                                            # Cache the pseudoinverse of the Laplacian, since it's the same regardless of the opinion polarization factor mu
      for factor in factors:                                                     # Loop over all possible values of opinion polarization mu
         sys.stderr.write(f"{factor}\t{out_p[i]}\t{conn_neigh}\t{run}...\n")
         os = make_os(125, factor, 8)                                            # Generate the opinion vector o depending on mu's value (and G's size)
         pol = ps.ge_multipolar(os, G, Q = Q)                                    # Calculate delta
         results.append((factor, out_p[i], conn_neigh, pol))

df = pd.DataFrame(data = results, columns = ("mu", "p_out", "n", "delta"))
df.to_csv(f"scores_mdim_run{run}.csv", index = False, sep = "\t")

