import pandas as pd
import networkx as nx

for net in ("obama", "guncontrol", "abortion", "vp_debate", "second_debate", "election"):
   G = nx.read_edgelist(f"{net}_edgelist.csv", delimiter = ',')
   node_pol = {}
   with open(f"{net}_user_scores.csv", 'r') as f:
      for line in f:
         fields = line.strip().split(',')
         node_pol[fields[0]] = float(fields[1])
   partition = [set(), set()]
   for node in node_pol:
      if node in G.nodes:
         if node_pol[node] > 0:
            partition[1].add(node)
         else:
            partition[0].add(node)
   print(f"{net} & {len(G.nodes)} & {len(G.edges)} & {nx.average_shortest_path_length(G):.3f} & {nx.transitivity(G):.3f} & {nx.algorithms.community.modularity(G, partition):.3f}\\\\")

for net in range(81, 117):
   G = nx.read_edgelist(f"congress{net}_edges.csv", delimiter = '\t', comments = 's', data = False)
   node_pol = pd.read_csv(f"congress{net}_nodes.csv", sep = "\t", dtype = {"icpsr": str})
   partition = [set(), set()]
   for index, row in node_pol.iterrows():
      if row["icpsr"] in G.nodes:
         if row["nominate_dim1"] > 0:
            partition[1].add(row["icpsr"])
         else:
            partition[0].add(row["icpsr"])
   print(f"{net} & {len(G.nodes)} & {len(G.edges)} & {nx.average_shortest_path_length(G):.3f} & {nx.transitivity(G):.3f} & {nx.algorithms.community.modularity(G, partition):.3f}\\\\")
