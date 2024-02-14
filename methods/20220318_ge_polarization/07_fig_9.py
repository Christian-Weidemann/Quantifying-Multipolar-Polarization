import numpy as np
import pandas as pd
import networkx as nx
from modules import ps
from scipy.integrate import odeint

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

# This function is the one we need to integrate over.
def f(o, t):
   return -L.dot(o)

G = nx.grid_2d_graph(4, 4)                           # Make the 4X4 grid graph
G = nx.convert_node_labels_to_integers(G)            # Conver its node labels to integer for convenience
L = np.array(nx.laplacian_matrix(G).todense())       # Calculate its Laplacian

nx.write_edgelist(G, "grid_graph", delimiter = "\t", data = False)

o = np.array([-1, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1])                 # Assign extreme opposite opinion scores to opposite corners of the grid graph
print(f"Delta: {ps.ge({n: o[n] for n in G.nodes}, {}, G, Q = ps._ge_Q(G)):.4f}")  # Calculate delta

t = list(range(13))                                         # This is the number of time steps we're going to use to integrate
df = pd.DataFrame(data = odeint(f, o, t)).T.reset_index()   # Make a dataframe with the solutions of the differential equation
df.to_csv("grid_graph_ts", sep = "\t", index = False)

# The following bins the results and writes them to file to generate the histograms
bins = pd.IntervalIndex.from_breaks(list([-(21 - _) / 20 for _ in range(42)]))
labels = [b.right for b in bins]

dfbins = pd.DataFrame(columns = ("bin",))
for i in t:
   dfi = pd.DataFrame()
   dfi[f"bin"] = pd.cut(df[i], bins).cat.rename_categories(labels)
   dfi = dfi.groupby(by = "bin").size().reset_index().rename(columns = {0: f"t{i}"})
   dfbins = dfbins.merge(dfi, on = "bin", how = "outer")

dfbins.to_csv("grid_graph_ts_hists", sep = "\t", index = False)

