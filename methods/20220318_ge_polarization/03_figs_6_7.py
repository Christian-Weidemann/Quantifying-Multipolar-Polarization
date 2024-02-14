import pandas as pd
import networkx as nx
from modules import ps

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

# This function makes the histograms for Figs 5 and 6
# It counts the number of users with a given opinion score, binned.
def make_hist(path):
    bins = pd.IntervalIndex.from_breaks(list([-(20 - _) / 20 for _ in range(41)]))
    labels = [(b.left + b.right) / 2 for b in bins]
    df = pd.read_csv(f"{path}_user_scores.csv", header = None, names = ("user", "pol"))
    df["pol"] = pd.cut(df["pol"], bins).cat.rename_categories(labels)
    df = df.groupby(by = "pol").size().reset_index()
    df.columns = ("pol", "count")
    df.to_csv(f"{path}_user_scores_hist.csv", index = False, sep = "\t")

# This function adds the opinion value to the edges
# This is the average opinion value of the two nodes connected by the edge
def add_weights(path):
    G = nx.read_edgelist(f"{path}_edgelist.csv", delimiter = ",")
    node_pol = {}
    with open(f"{path}_user_scores.csv", 'r') as f:
        for line in f:
            fields = line.strip().split(',')
            node_pol[fields[0]] = float(fields[1])
    nx.set_edge_attributes(G, {e: (node_pol[e[0]] + node_pol[e[1]]) / 2 for e in G.edges}, name = "pol")
    nx.write_edgelist(G, f"{path}_edgelist_wpol.csv", delimiter = ",", data = ("pol",))

# This function reads the edgelist and the opinion value
# It returns the delta polarization score
def calc_pol(path):
   G = nx.read_edgelist(f"{path}_edgelist.csv", delimiter = ',')
   o = {}
   with open(f"{path}_user_scores.csv", 'r') as f:
      for line in f:
         fields = line.strip().split(',')
         if fields[0] in G.nodes:
            o[fields[0]] = float(fields[1])
   return ps.ge(o, {}, G)

# Run the code for each of the networks
# vp_debate and second_debate might take a while due to their number of edges
for net in ("obama", "guncontrol", "abortion", "vp_debate", "second_debate", "election"):
   print(f"{net}: {calc_pol(f'{net}')}")
   add_weights(net)
   make_hist(net)
