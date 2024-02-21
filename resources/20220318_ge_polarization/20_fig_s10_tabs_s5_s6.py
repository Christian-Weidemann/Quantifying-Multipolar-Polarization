import pandas as pd
import graph_tool.all as gt
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# This function returns a 4x4 matrix with the edge counts across the 4 communities in the network.
def gt_edges(gt_state, color_list):
   edges = gt.adjacency(gt_state.get_bg(), gt_state.get_ers()).T                           # Get the adjacency matrix for the blocks
   edges = edges[edges.getnnz(1) > 0][:,edges.getnnz(0) > 0]                               # Remove all zeros rows/columns that the graph_tool library left in the data
   df = pd.DataFrame(edges.todense())                                                      # Convert to a pandas dataframe
   df.index = [color_list[sorted_block_pol_keys[sorted(blocks)[i]]] for i in df.index]     # Properly label the rows with the correct community color
   df.columns = [color_list[sorted_block_pol_keys[sorted(blocks)[i]]] for i in df.columns] # Properly label the columns with the correct community color
   return df.loc[color_list, color_list]                                                   # Return the dataframe with the rows and columns sorted in ascending opinion value

# This function returns a 4x4 matrix with the edge connection probabilities across the 4 communities in the network.
def gt_edge_probability(gt_state, color_list, sorted_block_pol_keys):
   sizes = {color_list[sorted_block_pol_keys[block]]: len(blocks[block]) for block in blocks}    # Get the size in number of nodes for each block
   edge_df = gt_edges(gt_state, color_list)                                                      # Get the 4x4 matrix of edge counts
   edge_df_prob = edge_df.copy()
   for column in color_list:                                                                     # For each block...
      for index in color_list:                                                                   # ...connecting to another block (even itself)...
         edge_df_prob[column][index] = (edge_df[column][index] / (sizes[column] * sizes[index])) # ..normalize the edge counts with the total number of possible edges between the blocks
   return edge_df_prob

# This function prints a latex table with the color-coded connection probabilities across the 4 blocks
def print_latex_table(df):
   cmap = plt.get_cmap("Reds")
   norm = colors.Normalize(vmin = df.min().min(), vmax = df.max().max())
   for index, row in df.iterrows():
      line = []
      line.append(f"{index}")
      for n in range(len(df.index)):
         color = cmap(norm(row[df.index[n]]))
         line.append("\cellcolor[HTML]{%02x%02x%02x}%1.4f" % (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255), row[df.index[n]]))
      print("%s\\\\" % " & ".join(line))

# This function prints a latex table with the color-coded average polarity for all blocks
def print_pol_table(df, block_pol, color_list, sorted_block_pol_keys):
   cmap = plt.get_cmap("RdBu_r")
   norm = colors.Normalize(vmin = -1, vmax = 1)
   for block in sorted_block_pol_keys:
      color = cmap(norm(block_pol[block]))
      print("%s & \cellcolor[HTML]{%02x%02x%02x}%1.2f\\\\" % (color_list[sorted_block_pol_keys[block]], int(color[0] * 255), int(color[1] * 255), int(color[2] * 255), block_pol[block]))

#read the input edgelist data
abortion_edgelist = pd.read_csv("abortion_edgelist.csv", names = ["source", "target"], dtype = str)

# Make a graph_tool object with the edgelist
g_abortion = gt.Graph(directed = False)
props = g_abortion.add_edge_list(abortion_edgelist.values, hashed = True)
# This is a dictionary connecting the graph_tool node id to the actual node id form the data
mapping_dict = {i: props[i] for i in range(g_abortion.num_vertices())}

# Infer the community structure specifying maximum 4 blocks (the algorithm is free to find fewer blocks)
abortion_state = gt.minimize_blockmodel_dl(g_abortion, multilevel_mcmc_args = {"B_max": 4})
abortion_state_comms = abortion_state.get_blocks()

# Load the users' opinion values
abortion_scores = pd.read_csv("abortion_user_scores.csv", header = None, names = ("user", "score")).set_index("user")

# Make a dictionary block id -> list of nodes in the block
blocks = {block: [int(mapping_dict[i]) for i in range(len(mapping_dict)) if abortion_state_comms.a[i] == block] for block in set(abortion_state_comms.a)}
# Make a dictionary block id -> avg node polarity in block
block_pol = {block: abortion_scores.loc[blocks[block]]["score"].mean() for block in blocks}
# Make a dictionary to sort the block ids so that block 0 is always the one with the lowest average opinion  and block 3 is the one with the highest
sorted_block_pol_keys = sorted(block_pol, key = block_pol.get)
sorted_block_pol_keys = {sorted_block_pol_keys[i]: i for i in range(4)}

# Write to file two columns: node id and the block id to which it belongs
with open("abortion_communities.csv", 'w') as f:
   f.write("node\tcomm\n")
   for i in range(g_abortion.num_vertices()):
      f.write(f"{mapping_dict[i]}\t{sorted_block_pol_keys[abortion_state_comms[i]]}\n")

# Block 0 = blue, block 1 = green, block 2 = purple, block 3 = red
color_list = ["blue", "green", "purple", "red"]
# Get connection probabilities between blocks
abortion_state_probs = gt_edge_probability(abortion_state, color_list, sorted_block_pol_keys)
# Print to standard output the latex table to be included in the paper
print_latex_table(abortion_state_probs)
print_pol_table(abortion_scores, block_pol, color_list, sorted_block_pol_keys)

