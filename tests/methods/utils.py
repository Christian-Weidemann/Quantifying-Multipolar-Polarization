import random
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import truncnorm

def ideo_make_p(p_out, n_comms, intercon):
   p = np.full((n_comms, n_comms), p_out)
   np.fill_diagonal(p, 0)                                                # Save the sum of p entries, this must be constant to ensure same expected # of edges
   for col in range(n_comms):                                              # For every column...
      for row in range(n_comms):                                           # ...and every row...
         if ((col < n_comms - intercon) and ((row - col) > intercon)) or ((row < n_comms - intercon) and ((col - row) > intercon)):     # ...find the entries that are k-1 steps away from the diagonal...
            p[row, col] = 0                                                # ...and set them to zero
   return p

def ideo_make_G(n_comms, nodes_per_comm, p):
   comm = nx.complete_graph(nodes_per_comm)
   G = nx.Graph()
   for _ in range(n_comms):
      G = nx.disjoint_union(G, comm)
   n_edges_to_rewire = (p * len(comm.edges)).round().astype(int)
   for c1 in range(n_comms):
      for c2 in range(n_comms):
         c1_bounds = c1 * nodes_per_comm, (c1 + 1) * nodes_per_comm
         c2_bounds = c2 * nodes_per_comm, (c2 + 1) * nodes_per_comm
         internal_c1_edges = [e for e in G.edges if e[0] >= c1_bounds[0] and e[0] < c1_bounds[1] and e[1] >= c1_bounds[0] and e[1] < c1_bounds[1]]
         internal_c1_edges = random.sample(internal_c1_edges, n_edges_to_rewire[c1, c2])
         c1_c2_edges = [(n1, n2) for n1 in list(range(c1_bounds[0], c1_bounds[1])) for n2 in list(range(c2_bounds[0], c2_bounds[1])) if not (n1, n2) in G.edges]
         c1_c2_edges = random.sample(c1_c2_edges, n_edges_to_rewire[c1, c2])
         G.remove_edges_from(internal_c1_edges)
         G.add_edges_from(c1_c2_edges)

   # nx.relabel_nodes(G, {n: n + 1 for n in G.nodes})  # Possible bug from an old version of the code
   return G

# The overall logic is that coomunities that are closer to each other have more similar opinions
# Current issue is that n and p_out are overwhelmed by shift, which seems to be the only things that matters
# One potential solution is to simplify the setup and set to zero all opinions outside the community
# But this might lose the multidimensional aspect (closer communities should have lower opinion difference)
def ideo_make_o(n_comms, nodes_per_comm, shift):
   o = pd.DataFrame()
   scale = 0.1
   loc_out = 0.5 - (shift / 2)
   loc_in = 0.5 + (shift / 2)
   # Making sure that both distributions are bound between 0 and 1
   a_out, b_out = (0 - loc_out) / scale, (1 - loc_out) / scale
   a_in, b_in = (0 - loc_in) / scale, (1 - loc_in) / scale
   for comm_id in range(n_comms):
      o[f"o{comm_id}"] = np.zeros(n_comms * nodes_per_comm)
      nodes_in_comm = np.in1d(range(o.shape[0]), range((comm_id * nodes_per_comm), ((comm_id + 1) * nodes_per_comm)))
      o[f"o{comm_id}"].values[~nodes_in_comm] = truncnorm.rvs(a_out, b_out, loc = loc_out, scale = scale, size = (n_comms - 1) * nodes_per_comm)
      o[f"o{comm_id}"].values[nodes_in_comm] = truncnorm.rvs(a_in, b_in, loc = loc_in, scale = scale, size = nodes_per_comm)
   return o
