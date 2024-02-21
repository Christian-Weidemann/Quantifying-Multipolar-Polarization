import random
import numpy as np
import networkx as nx
from modules import ps

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

# Function to create the o vector with a given opinion polarization factor (mu in the paper)
def make_o(size, factor):
   o = np.random.normal(size = size, loc = factor, scale = 0.2)
   o[o > 1] = 1 - (o[o > 1] - 1)
   o = np.concatenate([o, -o])
   o.sort()
   return o

### Left figure ###

# Lines 20 to 25 create the SBM and the o vector
probs = np.full((2, 2), 0.0024)
np.fill_diagonal(probs, 0.054)

G = nx.stochastic_block_model(sizes = [250] * 2, p = probs)
o = make_o(len(G.nodes) // 2, 0.2)
o = {n: o[n] for n in G.nodes}

block_dict = nx.get_node_attributes(G, "block") # Store in a dictionary the block to which each node belongs
# Identify the edges that are between the blocks, i.e. those established between nodes whose "block_dict" value is different
edges_between = [edge for edge in G.edges() if block_dict[edge[0]] != block_dict[edge[1]]]

# Initialize the list of polarity values, each entry in the list is the value of polarity at a given number of edges removed
# I.e. the 1st entry has 0 edges removed, the 2nd entry has 1 edge removed, 3rd entry has 2 edges removed, 4th entry 3, etc... 
pols = [ps.ge({n: o[n] for n in G.nodes}, {}, G, Q = ps._ge_Q(G)),]

while len(edges_between) > 0:
   edge = edges_between.pop(random.randrange(len(edges_between)))           # We pick one random edge between the blocks
   G.remove_edge(*edge)                                                     # And we remove it
   if nx.number_connected_components(G) > 1:                                # We might end up disconnecting the graph so let's check for it...
      break                                                                 # ...and interrupt the loop just in case, to avoid raising an error
   pols.append(ps.ge({n: o[n] for n in G.nodes}, {}, G, Q = ps._ge_Q(G)))   # Add the delta polarity value to the list

# Write output
with open("removeedges_2blocks.csv", 'w') as f:
   for i in range(len(pols)):
      f.write(f"{i}\t{pols[i]}\n")

### Right figure ###
G = nx.stochastic_block_model(sizes = [495, 5], p = [[0.1, 0], [0, 0.9]])      # Make an SBM with unbalanced isolated communities (495 nodes vs 5)
o = {n: 1 if n < 495 else -1 for n in G.nodes}                                 # Assign -1 polarity to one block and +1 to the other
G.add_edges_from([(490, 495), (491, 496), (492, 497), (493, 498), (494, 499)]) # Add a few edges between communities so that the graph has a single connected component

# Initialize the list of polarity values, each entry in the list is the value of polarity at a given number of nodes moved
# I.e. the 1st entry has 0 nodes moved from one block to the other, the 2nd entry has 1 node moved, 3rd entry has 2 nodes moved, 4th entry 3, etc... 
pols = [ps.ge(o, {}, G),]

for n in range(490):                                    # Move nodes in their id order (effectively random)
   degree = G.degree[n]                                 # Keep track of n's degree
   G.remove_node(n)                                     # Remove n from the networks (also removes all its edges)
   del o[n]                                             # Remove n from the o vector
   o[n + 500] = -1                                      # Since we're moving n from the +1 to the -1 community, the new (n+500) node must have a -1 opinion value in o
   for deg in range(degree):                            # We loop a number of times equal to n's old degree...
      G.add_edge(n + 500, random.randint(495, n + 499)) # And we add random edges for (n+500) to its new community mates (this might generate a lwoer degree if there aren't enough nodes to connect to)
   if nx.number_connected_components(G) > 1:            # It's possible that removing n disconnected some nodes in the +1 community, so we check for it...
      break                                             # ...and interrupt the loop just in case, to avoid raising an error
   pols.append(ps.ge(o, {}, G))                         # Add the delta polarity value to the list

# Write output
with open("balance_2blocks.csv", 'w') as f:
   for i in range(len(pols)):
      f.write(f"{i}\t{pols[i]}\n")

