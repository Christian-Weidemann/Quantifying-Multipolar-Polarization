import sys
import pandas as pd
import graph_tool.all as gt
from graph_tool import spectral
from collections import Counter
from scipy.stats import pearsonr, spearmanr

# This function estimates p_out by looking at the probability that an edge from a community leads outside the community
# Takes as input G (the graph) and comms (a graph_tool VertexPropertyMap with the community information)
def pout(G, comms):
   blocksizes = list(Counter(comms.a).values())    # comms.a is a vector with the community IID per node. A Counter will count how many nodes are in the community.
   pairs = blocksizes[0] * blocksizes[1]           # The number of possible edges between communities in an undirected graph
   A = spectral.adjacency(G)                       # The adjacency matrix of G
   edges = A[comms.a[:,None] != comms.a].sum() / 2 # The number of actual edges between communities (without double-counting since it is an undirected graph, hence dividing by 2)
   return edges / pairs                            # Returns p_out

mus_pouts = []
for net in range(81, 117):                                                                  # For each congress network...
   sys.stderr.write(f"{net}\n")                                                             # Read the edges
   df = pd.read_csv(f"../08_congress/congress{net}_edges.csv", dtype = str, sep = "\t")[["src", "trg"]].rename(columns = {"src": "source", "trg": "target"})
   G = gt.Graph(directed = False)                                                           # Transform into a graph_tool object
   props = G.add_edge_list(df.values, hashed = True)
   comms = gt.minimize_blockmodel_dl(G, multilevel_mcmc_args = {"B_max": 2}).get_blocks()   # Infer the communities
   df = pd.read_csv(f"../08_congress/congress{net}_nodes.csv", sep = "\t")                  # Get node information
   mean_right = df[df["nominate_dim1"] > 0]["nominate_dim1"].abs().mean()                   # Calculate the mean opinion of the right block
   mean_left = df[df["nominate_dim1"] < 0]["nominate_dim1"].abs().mean()                    # Calculate the mean opinion of the left block
   mus_pouts.append((f"congress{net}", (mean_right + mean_left) / 2, pout(G, comms)))       # Estimate mu, which is the average absolute value of the means of the two blocks, and p_out

df = pd.DataFrame(mus_pouts, columns = ("net", "mu", "pout"))
df.to_csv("mu_pout_congress.csv", sep = "\t", index = False)

print(pearsonr(df["mu"], df["pout"]))  # Calcualte the correlations between mu and p_out
print(spearmanr(df["mu"], df["pout"]))

# Repeat the above process for the Twitter networks
mus_pouts = []
for net in ("obama", "guncontrol", "abortion", "vp_debate", "second_debate", "election"):
   sys.stderr.write(f"{net}\n")
   df = pd.read_csv(f"{net}_edgelist.csv", names = ["source", "target"], dtype = str)
   G = gt.Graph(directed = False)
   props = G.add_edge_list(df.values, hashed = True)
   comms = gt.minimize_blockmodel_dl(G, multilevel_mcmc_args = {"B_max": 2}).get_blocks()
   df = pd.read_csv(f"{net}_user_scores.csv", header = None, names = ("user", "pol"))
   mean_right = df[df["pol"] > 0]["pol"].abs().mean()
   mean_left = df[df["pol"] < 0]["pol"].abs().mean()
   mus_pouts.append((net, (mean_right + mean_left) / 2, pout(G, comms)))

df = pd.DataFrame(mus_pouts, columns = ("net", "mu", "pout"))
df.to_csv("mu_pout_twitter.csv", sep = "\t", index = False)

print(pearsonr(df["mu"], df["pout"]))
print(spearmanr(df["mu"], df["pout"]))
