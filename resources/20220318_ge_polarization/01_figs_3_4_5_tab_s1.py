import sys
import numpy as np
import pandas as pd
import networkx as nx
from modules import ps
from modules import alt
from modules.rwc import rwc
from collections import defaultdict
from sklearn.neighbors import KernelDensity

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

# This function created the 2D KDE cell estimations, for plotting.
def kde2D(x, y, bandwidth, xbins = 100j, ybins = 100j, **kwargs): 
   xx, yy = np.mgrid[-1:1:xbins, -1:1:ybins]
   xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
   xy_train  = np.vstack([y, x]).T
   kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
   kde_skl.fit(xy_train)
   z = np.exp(kde_skl.score_samples(xy_sample))
   return xx, yy, np.reshape(z, xx.shape)

# Utility function to write the 2D KDE to a dataframe (and then to file)
def make_kde(df):
   xx, yy, zz = kde2D(df["pol"], df["avg_neighbor"], 0.1)
   df2 = pd.DataFrame()
   df2["x"] = xx.flatten()
   df2["y"] = yy.flatten()
   df2["z"] = zz.flatten()
   return df2

# This function creates the opinion vector o. It requires two inputs:
# - size: half the number of the nodes in the network;
# - factor: the absolute value of the average polarity of each side (mu in the paper)
# If factor = 0 there is no opinion polarizatio, with factor = 1 opinions cluster at +/- 1
def make_o(size, factor):
   o = np.random.normal(size = size, loc = factor, scale = 0.2) # Make a random normal distribution with std = 0.2
   o[o > 1] = 1 - (o[o > 1] - 1)                                # Mirror values out of the +/-1 bounds to be inbound
   o = np.concatenate([o, -o])                                  # Create the negative side of the distribution
   o.sort()                                                     # Sort, this will create community homophily in the SBM
   return {i: o[i] for i in range(o.shape[0])}                  # Transform into a dictionary, which si the input needed by the function

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
beta = 0.1              # Required for SIR model, probability of infection
gamma = 0.2             # Required for SIR model, recovery rate

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
         o = make_o(len(G.nodes) // 2, factor)                                   # Generate the opinion vector o depending on mu's value (and G's size)
         nx.set_node_attributes(G, o, "polar")                                   # Attach the opinion value to all nodes as an attribute -- required to calculate assortativity
         pol = ps.ge(o, {}, G, Q = Q)                                            # Calculate delta
         assort = nx.numeric_assortativity_coefficient(G, "polar")               # Calculate assortativity
         rwc_score = rwc(G)                                                      # Calculate RWC (relies on Garimella's original code -- translated from python2 to python3 by us)
         results.append((factor, out_p[i], conn_neigh, pol, assort, rwc_score))
         sir = alt.SIR(beta, gamma, G, o)                                        # Run the SIR simulation
         sir_binned = alt.bin_values_cinnelli(sir)                               # Bin the SIR results to generate comparable plots as those that can be found in Cinelli et al (2021)
         pd.DataFrame(data = sir_binned, columns = ("x", "y", "z", "a")).to_csv(f"sir_{factor}_{out_p[i]}_{conn_neigh}_{run}.csv", index = False, sep = "\t")
         G_avg_df = make_kde(alt.avg_neighbor_polarity(G, o))                    # Make the KDE of node opinion vs average neighbor opinion
         G_avg_df.to_csv(f"neighkde_{factor}_{out_p[i]}_{conn_neigh}_{run}.csv", index = False, sep = "\t")

df = pd.DataFrame(data = results, columns = ("mu", "p_out", "n", "delta", "assortativity", "rwc"))
df.to_csv(f"scores_run{run}.csv", index = False, sep = "\t")

