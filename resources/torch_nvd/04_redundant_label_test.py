import sys
sys.path.append("..")
import numpy as np
import pandas as pd
import networkx as nx
import torch_nvd as nvd

def write_distr(observed, expected):
   err = (observed - expected) / expected
   n_bins = 50
   bins = pd.interval_range(start = err.min() - 1e-9, end = err.max(), freq = (err.max() - err.min()) / n_bins)
   labels = [(b.left + b.right) / 2 for b in bins]
   err = pd.cut(err, bins)
   err = err.cat.rename_categories(labels)
   err = err.value_counts().reset_index()
   return err["index"], err["count"]

n_comms = 3
nodes_per_comm = 40

comm = nx.complete_graph(nodes_per_comm)

G = nx.Graph()
for _ in range(n_comms):
   G = nx.disjoint_union(G, comm)

for _ in range(n_comms - 1):
   G.add_edge(_ * nodes_per_comm, (_ + 1) * nodes_per_comm)

G = nx.relabel_nodes(G, {n: n + 1 for n in G.nodes})

o = pd.DataFrame()
for _ in range(n_comms):
   o[f"o{_}"] = np.zeros(nodes_per_comm * n_comms)
   o[f"o{_}"].values[(_ * nodes_per_comm):((_ + 1) * nodes_per_comm)] = np.ones(nodes_per_comm)

tensor = nvd.make_tensor(G, o)
Linv = nvd._Linv(tensor)

ideological_distance_perfect = nvd.ideological(tensor, Linv = Linv)
embedding_perfect = nvd._manifold(tensor, Linv)
ideological_distance_perfect_manifold = nvd.ideological_manifold(tensor, Linv = Linv, embedding = embedding_perfect)
tv_perfect = nvd.total_variation(tensor)

ideological_distance_extralabel = []
ideological_distance_extralabel_manifold = []
tvs = []
for factor in np.random.normal(loc = 0.5, scale = 0.01, size = 10000):
   o["o3"] = o["o1"] * factor
   tensor = nvd.make_tensor(G, o)
   ideological_distance_extralabel.append(nvd.ideological(tensor, Linv = Linv))
   embedding_extralabel = nvd._manifold(tensor, Linv)
   ideological_distance_extralabel_manifold.append(nvd.ideological_manifold(tensor, Linv = Linv, embedding = embedding_extralabel))
   tvs.append(nvd.total_variation(tensor))


df = pd.DataFrame()
df["ge_avg_value"], df["ge_avg_count"] = write_distr(ideological_distance_perfect, pd.Series(ideological_distance_extralabel))
df["ge_tsne_value"], df["ge_tsne_count"] = write_distr(ideological_distance_perfect_manifold, pd.Series(ideological_distance_extralabel_manifold))
df["ge_tv_value"], df["ge_tv_count"] = write_distr(tv_perfect, pd.Series(tvs))
df.to_csv("redundant_label_results.csv", sep = "\t", index = False)


print(f"TV Polscore Perfect: {tv_perfect}, TV Polscore Extralabel: {np.mean(tvs):.4f} ({np.std(tvs)})")

print(f"AVG Polscore Perfect: {ideological_distance_perfect}, AVG Polscore Extralabel: {np.mean(ideological_distance_extralabel):.4f} ({np.std(ideological_distance_extralabel)})")

print(f"tSNE Polscore Perfect: {ideological_distance_perfect_manifold}, tSNE Polscore Extralabel: {np.mean(ideological_distance_extralabel_manifold):.4f} ({np.std(ideological_distance_extralabel_manifold)})")


nodes_per_comm = 10

comm = nx.complete_graph(nodes_per_comm)

G = nx.Graph()
for _ in range(n_comms):
   G = nx.disjoint_union(G, comm)

for _ in range(n_comms - 1):
   G.add_edge(_ * nodes_per_comm, (_ + 1) * nodes_per_comm)

nx.write_edgelist(G, "G_redundant_label.txt", delimiter = "\t", data = False)

