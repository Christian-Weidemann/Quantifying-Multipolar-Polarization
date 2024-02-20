import torch, torch_geometric
import numpy as np
import pandas as pd
import networkx as nx
from scipy.special import binom
from sklearn.decomposition import PCA

device = "cuda" if torch.cuda.is_available() else "cpu"

def make_tensor(G, df):

   if len(G.nodes) < 2:
      raise ValueError("Graph must have at least two nodes")

   edge_index = [[], []]
   edge_attribute_names = list(list(G.edges(data = True))[0][2].keys())
   edge_attr = []
   for edge in G.edges(data = True):
      edge_index[0].append(edge[0])
      edge_index[1].append(edge[1])
      edge_index[0].append(edge[1])
      edge_index[1].append(edge[0])
      edge_attr.append([edge[2][edge_attribute_names[i]] for i in range(len(edge_attribute_names))])
   tensor = torch_geometric.data.Data(
      edge_index = torch.tensor(edge_index, dtype = torch.long).to(device),
      node_vects = torch.tensor(df.values, dtype = torch.float).double().to(device),
      edge_attr = torch.tensor(edge_attr, dtype = torch.float).double().to(device)
   )
   return tensor

def _Linv(tensor):
   L_ei, Lew = torch_geometric.utils.get_laplacian(tensor.edge_index)
   L = torch_geometric.utils.to_dense_adj(edge_index = L_ei, edge_attr = Lew)[0]
   return torch.linalg.pinv(L, hermitian = True).double()

def _er(tensor, Linv):
   if Linv is None:
      Linv = _Linv(tensor)
   pinv_diagonal = torch.diagonal(Linv)
   return pinv_diagonal.unsqueeze(0) +  pinv_diagonal.unsqueeze(1) - 2 * Linv

def _pairwise_distances(tensor, Linv):
   # I can only do a cholesky only if I can find a way to implement the sparse approximate algorithm
   # Cholesky is linear vs Linv superquadratic, so there are potential runtime gains here
   #u = torch.linalg.cholesky_ex(L)[0].double()
   if Linv is None:
      Linv = _Linv(tensor)
   distances = torch.zeros((tensor.node_vects.shape[1], tensor.node_vects.shape[1])).to(device)
   for i in range(tensor.node_vects.shape[1]):
      diff = tensor.node_vects[:,i] - tensor.node_vects[:,i + 1:].T
      distances[i,i + 1:] = (diff * torch.mm(Linv, diff.T).T).sum(dim = 1)
      # This is probably wrong because it expects diff to be a vector
      #distances[i,i + 1:] = (diff * torch.cholesky_solve(diff.unsqueeze(0).T, u.T).squeeze()).sum()
   return distances

def ge(src, trg, Linv = None):
   diff = src - trg
   return torch.sqrt((diff * torch.matmul(Linv, diff.T)).sum())

def ideological(tensor, Linv = None):
   distance_matrix = _pairwise_distances(tensor, Linv)
   return torch.sqrt(distance_matrix.sum() / binom(tensor.node_vects.shape[1], 2)).cpu().numpy().tolist()

def affective(tensor, Linv = None):
   src_ = tensor.node_vects[:,0] - torch.mean(tensor.node_vects[:,0])
   trg_ = tensor.node_vects[:,1] - torch.mean(tensor.node_vects[:,1])
   W = 1 / torch.exp(_er(tensor, Linv))
   numerator = (W * torch.outer(src_, trg_)).sum()
   denominator_src = torch.sqrt((W * torch.outer(src_, src_)).sum())
   denominator_trg = torch.sqrt((W * torch.outer(trg_, trg_)).sum())
   return (numerator / (denominator_src * denominator_trg)).cpu().numpy()

def affective_noline(tensor, Linv = None):
   if Linv is None:
      Linv = _Linv(tensor)
   nodes_1d = _manifold(tensor, Linv)
   nodes_1d = (nodes_1d - nodes_1d.min()) / (nodes_1d.max() - nodes_1d.min())
   y = torch.abs(nodes_1d.unsqueeze(1) - nodes_1d).squeeze(2)
   y = y[tensor.edge_index[0][::2], tensor.edge_index[1][::2]]
   W = 1 / torch.exp(_er(tensor, Linv))
   W = W[tensor.edge_index[0][::2], tensor.edge_index[1][::2]]
   W_sum = W.sum()
   x_ = tensor.edge_attr[0] - torch.mean(W * tensor.edge_attr[0])
   y_ = y - ((W * y).sum() / W_sum)
   numerator = (W * x_ * y_).sum() / W.sum()
   denominator_x = (W * (x_ ** 2)).sum() / W.sum()
   denominator_y = (W * (y_ ** 2)).sum() / W.sum()
   return (numerator / torch.sqrt(denominator_x * denominator_y)).cpu().numpy()

def _manifold(tensor, Linv):
   ideology_distances = _pairwise_distances(tensor, Linv).double().cpu().numpy()
   ideology_distances = ideology_distances + ideology_distances.T
   # PCA is more stable than TSNE and actually solves the dimension problem of correlated ideologies
   # However it seems to find difficult to space properly the nodes in the space, the distributions are much more questionable than TSNE
   reducer = PCA(n_components = 1)
   embedding = reducer.fit_transform(ideology_distances)
   embedding = torch.tensor(embedding).double().to(device)
   return torch.mm(tensor.node_vects, embedding)

def ideological_manifold(tensor, Linv = None, embedding = None):
   if Linv is None:
      Linv = _Linv(tensor)
   if embedding is None:
      embedding = _manifold(tensor, Linv)
   embedding = (embedding - embedding.min()) / (embedding.max() - embedding.min())
   return torch.sqrt(torch.mm(embedding.T, torch.mm(Linv, embedding))).cpu().numpy()[0][0]

# This is from Martin-Gutierrez et al. 2023.
# As is, it is simplistic and ignores G. But it makes full use of the opinion data we have.
# Alternatively, we can throw out most of o and use G to infer opinion scores rather than using the actual ones.
def total_variation(tensor):
   norm_factor = tensor.node_vects.shape[0] / tensor.node_vects.shape[1]
   return torch.trace(torch.cov(tensor.node_vects)).cpu().numpy() / norm_factor # figure out how to normalize so that it gets more evenly spread between 0 and 1

