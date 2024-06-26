"""
A modified and expanded version of /resources/torch_nvd/utils.py, originally provided by Michele Coscia. 
"""

import random
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import truncnorm
import seaborn as sns
from matplotlib.colors import ListedColormap, rgb2hex
import matplotlib.pyplot as plt
import os

FIGURES_PATH = "figures/"

def cmap(cmap_type, n=10, as_colormap=True, shuffle=True):

   palette_tab10 = sns.color_palette("tab10", n)
   blue = '#1f77b4'
   red = '#d62728'
   grey = '#d9d2dd'
   green = '#2ca02c'


   if cmap_type == "bgr":
      colors = [blue, green, red]
      cmap = sns.blend_palette(colors, n_colors=len(colors), as_cmap=as_colormap)

   elif cmap_type == "bpr":
      purple = rgb2hex(sns.blend_palette([blue, red], as_cmap=True)(0.5))
      colors = [blue, purple, red]
      cmap = sns.blend_palette(colors, n_colors=len(colors), as_cmap=as_colormap)
   
   elif cmap_type == "br":
      colors = [blue, grey, red]
      cmap = sns.blend_palette(colors, n_colors=len(colors), as_cmap=as_colormap)

   elif cmap_type == "qualitative":
      cmap = list(palette_tab10.as_hex())
      if shuffle:
         random.shuffle(cmap)
      if as_colormap:
         cmap = ListedColormap(cmap)

   elif cmap_type == "colors":
      cmap = [blue, red, green, grey]
         
   else:
      raise ValueError("Invalid cmap_type")
   
   return cmap

def share_axes(axes, sharex=True, sharey=True):
   """
   Share axes in a grid of subplots.
   from https://stackoverflow.com/a/77862871
   """
   if isinstance(axes, np.ndarray):
        axes = axes.flat  # from plt.subplots
   elif isinstance(axes, dict):
        axes = list(axes.values())  # from plt.subplot_mosaic
   else:
        axes = list(axes)
   ax0 = axes[0]
   for ax in axes:
      if sharex:
         ax.sharex(ax0)
         if not ax.get_subplotspec().is_last_row():
               ax.tick_params(labelbottom=False)
      if sharey:
         ax.sharey(ax0)
         if not ax.get_subplotspec().is_first_col():
               ax.tick_params(labelleft=False)


def save_figure(figure_name, figure_folder=None, overwrite=False, dpi=600):

   if figure_folder is None:
      figure_path = FIGURES_PATH + figure_name
   else:
      folder_path = FIGURES_PATH + figure_folder
      if not os.path.exists(folder_path):
         os.makedirs(folder_path)
         print(f"Folder created at {folder_path}")
      figure_path = folder_path + figure_name

   if overwrite or not os.path.exists(figure_path):
      plt.savefig(figure_path, bbox_inches="tight", dpi=dpi)
      print("Figure saved")
   else:
      print("Figure already exists")


def ideo_make_p(p_out, n_comms, intercon):
   """
   Provided by Michele Coscia.
   """
   p = np.full((n_comms, n_comms), p_out)
   np.fill_diagonal(p, 0)                                                # Save the sum of p entries, this must be constant to ensure same expected # of edges
   for col in range(n_comms):                                              # For every column...
      for row in range(n_comms):                                           # ...and every row...
         if ((col < n_comms - intercon) and ((row - col) > intercon)) or ((row < n_comms - intercon) and ((col - row) > intercon)):     # ...find the entries that are k-1 steps away from the diagonal...
            p[row, col] = 0                                                # ...and set them to zero
   return p

def ideo_make_G(n_comms, nodes_per_comm, p):
   """
   Provided by Michele Coscia.
   """
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

   return G

def stochastic_block_model(n_comms, nodes_per_comm, p_in, p_out, community_labels = True, positions = True):

   p = np.full((n_comms, n_comms), p_out)
   np.fill_diagonal(p, p_in)

   G = nx.stochastic_block_model(sizes = [nodes_per_comm] * n_comms, p = p)
   
   # Ensure that the graph is connected
   retry = 0
   while nx.number_connected_components(G) > 1:                                 
      G = nx.stochastic_block_model(sizes = [nodes_per_comm] * n_comms, p = p)
      retry += 1
      if retry > 100:
         raise Exception("Could not generate a connected graph")

   # Add community labels to nodes
   if community_labels:
      for i in range(n_comms):
         for node in range(i * nodes_per_comm, (i + 1) * nodes_per_comm):
            G.nodes[node]["community"] = i
   
   # Add node positions as attributes
   if positions:
      pos = nx.kamada_kawai_layout(G)
      nx.set_node_attributes(G, pos, 'pos')

   return G

def make_community_graph(n_comms, nodes_per_comm):
   comm = nx.complete_graph(nodes_per_comm)
   G = nx.Graph()
   for _ in range(n_comms):
      G = nx.disjoint_union(G, comm)

   components = list(nx.connected_components(G))

   # add a random edge between each pair of components
   # ensuring that no node is on more than one added edge
   for i in range(len(components)):
      G.add_edge(max(components[i]), min(components[(i + 1) % len(components)]))

   return G



# The overall logic is that coomunities that are closer to each other have more similar opinions
# Current issue is that n and p_out are overwhelmed by shift, which seems to be the only things that matters
# One potential solution is to simplify the setup and set to zero all opinions outside the community
# But this might lose the multidimensional aspect (closer communities should have lower opinion difference)
def ideo_make_o(n_comms, nodes_per_comm, shift):
   """
   Provided by Michele Coscia.
   """
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
