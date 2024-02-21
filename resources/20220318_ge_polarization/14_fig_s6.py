import random, sys
import numpy as np
import pandas as pd
import networkx as nx
from modules import ps
from modules import alt
from modules.rwc import rwc
from scipy.stats import sem
from sklearn.neighbors import KernelDensity

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

# Following functions are documented in 01_figs_3_4_5_tab_s1.py
# Only difference is "make_o" which sets loc = 0 rather than taking it as aparameter, and takes "scale" as a parameter to set the std of the o vector.
def kde2D(x, y, bandwidth, xbins = 100j, ybins = 100j, **kwargs): 
   xx, yy = np.mgrid[-1:1:xbins, -1:1:ybins]
   xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
   xy_train  = np.vstack([y, x]).T
   kde_skl = KernelDensity(bandwidth = bandwidth, **kwargs)
   kde_skl.fit(xy_train)
   z = np.exp(kde_skl.score_samples(xy_sample))
   return xx, yy, np.reshape(z, xx.shape)

def make_kde(df):
   xx, yy, zz = kde2D(df["pol"], df["avg_neighbor"], 0.1)
   df2 = pd.DataFrame()
   df2["x"] = xx.flatten()
   df2["y"] = yy.flatten()
   df2["z"] = zz.flatten()
   return df2

def make_o(size, scale):
   o = np.random.normal(size = size, loc = 0, scale = scale)
   o[o > 1] = 1 - (o[o > 1] - 1)
   o = np.concatenate([o, -o])
   o.sort()
   return o

beta = 0.1
gamma = 0.2

G = nx.ring_of_cliques(2, 50)

Q = ps._ge_Q(G)

df = []
bins = [(x - 20) / 20 for x in range(41)]
bin_labels = [(x - 19.5) / 19.5 for x in range(40)]
o_distrs = pd.DataFrame(index = bin_labels)

index1 = list(range(50))
index2 = list(range(50, 100))
np.random.shuffle(index1)
np.random.shuffle(index2)
o_fig = pd.DataFrame(index = index1 + index2)

# Repeat the experiments of 01_figs_3_4_5_tab_s1.py, but this time only varying the standard deviation of o
# Refer to that script for the full documentation of these code lines
for stdev in (0.2, 0.05, 0.001):
   sys.stderr.write(f"{stdev}\n")
   rwcs = []
   ges = []
   assorts = []
   os = np.array([])
   for run in range(25): # Make 25 independent runs
      o = make_o(len(G.nodes) // 2, stdev)
      o_fig[stdev] = o
      os = np.concatenate([os, o])
      o = {n: o[n] for n in G.nodes}
      nx.set_node_attributes(G, o, "polar")
      rwcs.append(rwc(G))
      assorts.append(nx.numeric_assortativity_coefficient(G, "polar"))
      ges.append(ps.ge(o, {}, G, Q = Q))
   o_distrs[stdev] = pd.cut(os, bins, labels = bin_labels).value_counts()
   df.append((
      stdev,
      np.mean(assorts),
      np.std(assorts),
      np.mean(rwcs),
      np.std(rwcs),
      np.mean(ges),
      np.std(ges),
   ))
   sir = alt.SIR(beta, gamma, G, o)
   sir_binned = alt.bin_values_cinnelli(sir)
   pd.DataFrame(data = sir_binned, columns = ("x", "y", "z", "a")).to_csv(f"sir_{stdev}.csv", index = False, sep = "\t")
   G_avg_df = make_kde(alt.avg_neighbor_polarity(G, o))
   G_avg_df.to_csv(f"neighkde_{stdev}.csv", index = False, sep = "\t")
   nx.set_edge_attributes(G, {e: (o_fig.loc[e[0]][stdev] + o_fig.loc[e[1]][stdev]) / 2 for e in G.edges}, str(stdev))

df = pd.DataFrame(data = df, columns = ("stdev", "assort", "assort_std", "rwc", "rwc_std", "ge", "ge_std"))
df.to_csv("measures.csv", index = False, sep = "\t")

o_distrs.reset_index().to_csv("o_distrs.csv", index = False, sep = "\t")
o_fig.reset_index().to_csv("example_os.csv", index = False, sep = "\t")

nx.write_edgelist(G, "network.tsv", delimiter = "\t", data = ["0.2", "0.05", "0.001"])

