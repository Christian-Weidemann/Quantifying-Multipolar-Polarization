import random, sys
import numpy as np
import pandas as pd
import networkx as nx
from modules import ps
from scipy.linalg import expm

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

# Calculate the time t in which the system reaches an equilibrium (i.e. the stdev of o is below some arbitrary low number).
def time_to_equilibrium(G, o):
   L = np.array(nx.laplacian_matrix(G).todense()) # Get the Laplacian of G
   K = expm(-L * 1)                               # Calculate the matrix exponentiation, fixing heat capacity at 1.
   step = 0                                       # This variable keeps track of how many steps we took.
   while np.std(o) > 2e-4:                        # We decided to use epsilon = 2e-4. This value doesn't matter, provided it is small enough. Lower values take more time to converge.
      o = K.dot(o)                                # Make one diffusion step, diffusing the current o values to new o values via the kernel K.
      step += 1                                   # Increment the number of time steps.
   return step, o

# Function to modify the p hyperparameter of the SBM. Check 01_figs_2_3_4_tab_s1.py comments for a full description.
def update_p(p, k):
   p_sum = p.sum()
   for col in range(8):
      for row in range(8):
         if ((col < 8 - k) and ((row - col) > k)) or ((row < 8 - k) and ((col - row) > k)):
            p[row, col] = 0
   p *= (p_sum / p.sum())
   return p

# Function to modify the o values depending on the opinion polarization factor (mu in the paper). Check 01_figs_2_3_4_tab_s1.py comments for a full description.
def make_o(size, factor):
   o = np.random.normal(size = size, loc = factor, scale = 0.2)
   o[o > 1] = 1 - (o[o > 1] - 1)
   o = np.concatenate([o, -o])
   o.sort()
   return o

factors = (0.0, 0.2, 0.4, 0.6, 0.8)               # All testes mu values
in_p =  ( 0.039,  0.054,  0.062,  0.064,  0.067)  # All tested p_in values
out_p = (0.0042, 0.0024, 0.0012, 0.0006, 0.0003)  # All tested p_out values

i = int(sys.argv[1])                # Command line parameter that selects the p_in and p_out values (should be between 0 and 3)
j = int(sys.argv[2])                # Command line parameter that selects the mu value (should be between 0 and 4)
k = int(sys.argv[3])                # Command line parameter that selects the n value (should be between 1 and 7)
sys.stderr.write(f"{i} {j} {k}\n")

with open(f"pol_{i}_{j}_{k}.csv", 'w') as f:
   f.write("opinion_pol\tout_p\tneigh_comms\tequilibrium_t\tpolarization\n")
   for _ in range(10):                                               # Loop 10 times for 10 independent initializations
      probs = np.full((8, 8), out_p[i])                              # Lines 51 to 58 create the SBM and the o vector
      np.fill_diagonal(probs, in_p[i])
      probs = update_p(probs, k)
      G = nx.stochastic_block_model(sizes = [125] * 8, p = probs)
      while nx.number_connected_components(G) > 1:
         G = nx.stochastic_block_model(sizes = [125] * 8, p = probs)
      Q = ps._ge_Q(G)
      o = make_o(len(G.nodes) // 2, factors[j])
      pol = ps.ge({n: o[n] for n in G.nodes}, {}, G, Q = Q)          # Calculate delta polarity of the system
      t, o = time_to_equilibrium(G, o)                               # Calculate the time step t in which o reaches a heat diffusion equilibrium
      f.write(f"{factors[j]}\t{out_p[i]}\t{k}\t{t}\t{pol}\n")        # Write output
      f.flush()



