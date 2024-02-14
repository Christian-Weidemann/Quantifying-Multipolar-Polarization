import networkx as nx
from modules import ps

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

for n in range(2, 251):               # n is the number of nodes in the path graph
   G = nx.path_graph(n)               # Generate a path graph
   o = {0: -1, n - 1: 1}              # Create the opinion vector o as a dictionary, with -1 in one extreme of the chain and +1 in the other extreme, 0 otherwise (implied in the dictionary)
   print(f"{n}\t{ps.ge(o, {}, G)}")   # Print delta polarization, which is equal to sqrt(n - 1), because a path graph of n nodes has n-1 edges.
