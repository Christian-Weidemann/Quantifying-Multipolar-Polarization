import networkx as nx
import numpy as np
from scipy.special import binom
from scipy.sparse import csgraph

# Function to calculate the pseudoinverse of the Laplacian of the network
def _ge_Q(network):
    A = nx.adjacency_matrix(network).todense().astype(float)
    return np.linalg.pinv(csgraph.laplacian(np.matrix(A), normed=False))

def ge(src, trg, network, Q=None):
    """Calculate GE for network.

    Parameters:
    ----------
    srg: vector specifying node polarities
    trg: vector specifying node polarities
    network: networkx graph
    Q: pseudoinverse of Laplacian of the network
    """
    if nx.number_connected_components(network) > 1:
        raise ValueError("""Node vector distance is only valid if calculated on a network with a single connected component.
                       The network passed has more than one.""")
    src = np.array([src[n] if n in src else 0. for n in network.nodes()])
    trg = np.array([trg[n] if n in trg else 0. for n in network.nodes()])
    diff = src - trg
    if Q is None:
        Q = _ge_Q(network)

    ge_dist = diff.T.dot(np.array(Q).dot(diff))

    if ge_dist < 0:
        ge_dist = 0

    return np.sqrt(ge_dist)

def resistance(network):
    """Calculate effective resistance for each node pair in the network.

    Parameters:
    ----------
    network: networkx graph.
    """

    n = len(network.nodes)
    L = csgraph.laplacian(np.matrix(nx.adjacency_matrix(network).todense().astype(float)), normed=False)
    Phi = np.ones((n, n)) / n
    Gamma = np.linalg.pinv(L + Phi)

    # calculate resistance for all node pairs
    res = np.array(
        [[Gamma[i, i] + Gamma[j, j] - (2 * Gamma[i, j]) if i != j else 0 for j in range(n)] for i in range(n)])

    return res

def ge_multipolar(os, network, Q = None):
    """Calculate multipolar GE for network.

    Parameters:
    ----------
    os: vector specifying all node polarities
    network: networkx graph
    Q: pseudoinverse of Laplacian of the network
    """
    if nx.number_connected_components(network) > 1:
        raise ValueError("""Node vector distance is only valid if calculated on a network with a single connected component.
                       The network passed has more than one.""")
    os = [np.array([os[n][i] for n in network.nodes()]) for i in range(os[0].shape[0])]
    if Q is None:
        Q = _ge_Q(network)
    conflict_sum = 0
    for i in range(len(os) - 1):
       for j in range(i + 1, len(os)):
          diff = os[i] - os[j]
          conflict_sum += diff.T.dot(np.array(Q).dot(diff))

    ge_dist = conflict_sum / binom(len(os), 2)

    if ge_dist < 0:
        ge_dist = 0

    return np.sqrt(ge_dist)
