import sys, utils
sys.path.append("..")
import numpy as np
import pandas as pd
from collections import defaultdict
import torch_nvd as nvd

def odd_ratio(p):
   edges_in_community = (nodes_per_comm * (nodes_per_comm - 1)) / 2                    # Number of edges in a clique of n nodes is n(n-1)/2.
   n_edges_rewired = edges_in_community * p * (n_comms - 1)                            # p tells the prob of rewiring an edge to a given community, so we multiply it with the 
                                                                                       # available edges and the number of communities it can go to (since we repeat the process per 	
                                                                                       # community). This gives us the total number of edges moving from the community for a p.
   p_in = (edges_in_community - n_edges_rewired) / edges_in_community                  # The p of an edge to stay in its community is the number of edges that did not move over the 
                                                                                       # total possible number of edges.
   p_out = (2 * n_edges_rewired) / (((n_comms - 1) * nodes_per_comm) * nodes_per_comm) # The p of an edge to move to another community is the number of edges we rewired over all 
                                                                                       # possible edges that can exist (#nodes outside the community times #nodes inside a community. 
                                                                                       # Multiplied by 2 because each community sends AND receives the same number of edges overall.
   return p_out / p_in

n_runs = 100
n_comms = 6
nodes_per_comm = 40
p_outs = 1 / np.geomspace(6, 780, 5).round().astype(int)
shifts = [0.0, 0.25, 0.5, 0.75, 1.0]
intercons = [5, 4, 3, 2, 1]

o_s = defaultdict(list)
results = []
for pi in range(len(p_outs)):
   for shift in shifts:
      for intercon in intercons:
         for _ in range(n_runs):
            sys.stderr.write(f"                                                                    \r")
            sys.stderr.write(f"pi: {p_outs[pi]:.4f}\tshift: {shift}\tintercon: {intercon}\trun: {_}\r")
            p = utils.ideo_make_p(p_outs[pi], n_comms, intercon)
            G = utils.ideo_make_G(n_comms, nodes_per_comm, p)
            o = utils.ideo_make_o(n_comms, nodes_per_comm, shift)
            tensor = nvd.make_tensor(G, o)
            Linv = nvd._Linv(tensor)
            ge_avg = nvd.ideological(tensor, Linv = Linv)
            ge_tsne = nvd.ideological_manifold(tensor, Linv = Linv)
            tv = nvd.total_variation(tensor)
            # Transform probability into odds ratio
            results.append((odd_ratio(p_outs[pi]), shift, intercon, ge_avg, ge_tsne, tv))
            if pi == 0 and intercon == 5:
               o_s[shift].append(o)

sys.stderr.write('\n')
            
o_s = {shift: pd.concat(o_s[shift]) for shift in o_s}

n_bins = 50
bins = pd.interval_range(start = 0.0, end = 1.0, freq = 1 / n_bins)
labels = (np.array(list(range(n_bins))) / n_bins) + (1 / (2 * n_bins))
for shift in o_s:
   o_plot = pd.DataFrame(index = labels)
   for column in o.columns:
      o_s[shift][column] = pd.cut(o_s[shift][column], bins)
      o_s[shift][column] = o_s[shift][column].cat.rename_categories(labels)
      o_plot[column] = o_s[shift].groupby(by = column).size()
   o_plot = o_plot.fillna(0.0) / n_runs
   o_plot.reset_index().to_csv(f"shift_{shift}_nodes_distrs.txt", sep = "\t", index = False)

results = pd.DataFrame(data = results, columns = ("p_out", "shift", "intercon", "ge_avg", "ge_tsne", "tv"))
results.to_csv("ideological_polarization.csv", sep = "\t", index = False)
