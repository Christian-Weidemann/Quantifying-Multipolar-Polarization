"""
A modified and expanded version of /resources/torch_nvd/torch_nvd.py, originally provided by Michele Coscia. 
"""

import torch, torch_geometric
import numpy as np
import pandas as pd
import networkx as nx
from scipy.special import binom
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import warnings

device = "cuda" if torch.cuda.is_available() else "cpu"


def make_tensor(G, df):

   if len(G.nodes) < 2:
      raise ValueError("Graph must have at least two nodes")
   
   # Normalize opinions (columns of df) to sum to 1, fill NaNs with 0 (the case of zero-sum column)
   df = df.div(df.sum(), axis=1).fillna(0)

   # Normalize rows of df
   # df = df.div(df.sum(axis = 1), axis = 0)

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
   """
   Compute the pseudo-inverse of the Laplacian.
   Provided by Michele Coscia.
   """
   L_ei, Lew = torch_geometric.utils.get_laplacian(tensor.edge_index)
   L = torch_geometric.utils.to_dense_adj(edge_index = L_ei, edge_attr = Lew)[0]
   return torch.linalg.pinv(L, hermitian = True).double()


def _er(tensor, Linv):
   """
   Compute the effective resistance matrix.
   Provided by Michele Coscia.
   """
   if Linv is None:
      Linv = _Linv(tensor)
   pinv_diagonal = torch.diagonal(Linv)
   return pinv_diagonal.unsqueeze(0) +  pinv_diagonal.unsqueeze(1) - 2 * Linv


def _pairwise_distances(tensor, Linv):
   """
   Compute the pairwise GE distance between each pair of opinion dimensions.
   Provided by Michele Coscia.
   """
   distances = torch.zeros((tensor.node_vects.shape[1], tensor.node_vects.shape[1])).to(device)
   for i in range(tensor.node_vects.shape[1]):
      diff = tensor.node_vects[:,i] - tensor.node_vects[:,i + 1:].T
      distances[i,i + 1:] = (diff * torch.mm(Linv, diff.T).T).sum(dim = 1)
   return distances


def ge(src, trg, Linv):
   """
   Generalized Euclidean distance between src and trg vectors of node opinions.
   Considers graph topology from the pseudo-inverse of the Laplacian, Linv.
   """
   diff = src - trg
   # if diff has dimension other than 1, use diff.mT, else use diff.T
   if len(diff.shape) == 1:
      diff_T = diff.T
   else:
      diff_T = diff.mT
   return torch.sqrt((diff * torch.matmul(Linv, diff_T)).sum())


def pairwise_average(tensor):
   """
   Average GE distance between each pair of opinion dimensions.
   Provided by Michele Coscia.
   """
   Linv = _Linv(tensor)
   distance_matrix = _pairwise_distances(tensor, Linv)
   # print(distance_matrix.sum(), "\n")
   return torch.sqrt(distance_matrix.sum() / binom(tensor.node_vects.shape[1], 2)).cpu().numpy().tolist()


def _manifold(tensor, embedding):

   Linv = _Linv(tensor)

   # Converting the embedding to a tensor and moving it to the device
   embedding = torch.tensor(embedding).double().to(device)  

   # Compute the GE distance between the above- and below-mean sides of the embedding (?)
   dist = torch.sqrt(torch.mm(embedding.T, torch.mm(Linv, embedding))).cpu().numpy()[0][0]

   return dist

def PC_manifold(tensor):
   opinion_matrix = tensor.node_vects.cpu().numpy()
   reducer = PCA(n_components = 1, svd_solver = "arpack")
   embedding = reducer.fit_transform(opinion_matrix)
   return _manifold(tensor, embedding)
   
def MDS_manifold(tensor):
   opinion_matrix = tensor.node_vects.cpu().numpy()
   # Ignoring warnings while fitting MDS models
   with warnings.catch_warnings():  
      warnings.simplefilter('ignore') 
      reducer = MDS(n_components = 1, n_init = 1, dissimilarity = "euclidean")
      embedding = reducer.fit_transform(opinion_matrix)
   return _manifold(tensor, embedding)


def avg_dist_to_mean(tensor):
   """
   Average GE distance between opinion dimensions and the mean opinion across opinion dimensions.
   1. Compute the mean opinion across opinion dimensions.
   2. Compute the GE distance between each opinion dimension and the mean opinion using ge().
   3. Return the average of the GE distances.
   """
   Linv = _Linv(tensor)
   mean_opinion = torch.mean(tensor.node_vects, dim = 1)
   dists = torch.zeros(tensor.node_vects.shape[1])
   for i in range(tensor.node_vects.shape[1]):
      dists[i] = ge(tensor.node_vects[:,i], mean_opinion, Linv)
   return torch.mean(dists).cpu().numpy()


def total_variation(tensor, Linv = None):
   """
   This is from Martin-Gutierrez et al. 2023.
   As is, it is simplistic and ignores G. But it makes full use of the opinion data we have.
   Alternatively, we can throw out most of o and use G to infer opinion scores rather than using the actual ones.
   Provided by Michele Coscia.
   """
   norm_factor = tensor.node_vects.shape[0] / tensor.node_vects.shape[1]
   return torch.trace(torch.cov(tensor.node_vects)).cpu().numpy() / norm_factor
