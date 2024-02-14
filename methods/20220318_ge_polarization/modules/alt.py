import pandas as pd
from statistics import mean
from scipy.stats import sem
import numpy as np
import networkx as nx

# The functions below implement two alternative methods to measure political polarization proposed by
# Cinnelli et al. (2020): https://doi.org/10.1073/pnas.2023301118

def avg_neighbor_polarity(network, node_dict):
    """Calculate the average neighbor polarity for all nodes in the network.

    Parameters:
    ----------
    network: networkx graph
    node_dict: dictionary specifying the polarity score of each node
    """
    # nx.isolates() is a generator and we check if the generator is empty (= there are no isolates)
    _empty = object()

    if next(nx.isolates(network), _empty) is _empty:

        # create a list of tuples with the following info: (node polarity, avg polarity of neighbors of this node)
        avg_neigh_pol = []
        for node in network:
            avg_neigh_pol.append((node_dict[node], mean([node_dict[neigh] for neigh in network.neighbors(node)])))

    else:
        raise ValueError("There are isolate nodes in the network. Since they don't have any neighbors, it is not possible to calculate their average neighbor polarity.")

    return pd.DataFrame(avg_neigh_pol, columns=["pol", "avg_neighbor"])


def SIR(beta, gamma, network, node_dict):
    """Simulate SIR diffusion for a given network.

    Parameters:
    ----------
    beta: infection probability
    gamma: recovery probability
    network: networkx graph
    node_dict: dictionary specifying the polarity score of each node
    """

    rng = np.random.default_rng()

    # beta is the beta parameter passed to the function multiplied by 1/avg degree of the network
    beta = beta * (1/mean(dict(network.degree()).values()))

    # list to write results to
    results = []

    for seed_node in network.nodes():

        inf = set()  # infected
        rec = set()  # recovered

        # start the process with seed node and end once there are no infected nodes left
        inf.add(seed_node)

        while inf:

            # get all neighbors of the infected nodes
            for node in inf.copy():
                for neigh in set(network.neighbors(node)):

                    # decide with probability beta whether node should be added to the set of infected nodes
                    if neigh not in inf:
                        if neigh not in rec:
                            if rng.random() < beta:
                                inf.add(neigh)

                # decide with probability gamma if node is not infected any longer
                if rng.random() < gamma:
                    inf.remove(node)

                    # if the node in question is not the seed node, then add it to the set of recovered nodes
                    if node != seed_node:
                        rec.add(node)

        # if there is a set of recovered nodes (there isn't always, this depends on beta and gamma)
        # get the seed node leaning and the average leaning for the set of recovered users and save as tuple
        if rec:
            seed_leaning = node_dict[seed_node]
            avg_leaning = mean([node_dict[user] for user in rec])
            results.append((seed_leaning, avg_leaning, len(rec)))

    return results


def bin_values_boxplot(results):
    """Bin the values returned by the SIR simulation for plotting boxplots

    Parameters:
    ----------
    results: list containing the SIR results as tuples of format: (seed_leaning, avg_leaning, len(rec)))
    """

    # define the number of bins and the number of values in each bin
    number_of_bins = round((np.sqrt(len(results)) / 3))
    values_per_bin = int(len(results) / number_of_bins)

    # create a list "nest" that contains a sublist with binned, consecutive values
    nest = [sorted(results)[i:i + values_per_bin] for i in range(0, number_of_bins * values_per_bin, values_per_bin)]
    plot_list = []

    for i, _ in enumerate(nest):
        if len(nest[i]) == values_per_bin:
            seed_mean = mean([ele[0] for ele in nest[i]])
            influence_mean = [ele[1] for ele in nest[i]]
            influence_size = mean([ele[2] for ele in nest[i]])

            plot_list.append((seed_mean, influence_mean, influence_size))

    return plot_list

def bin_values_cinnelli(results):
    """Bin the values returned by the SIR simulation for plotting that looks similar to plots shown in
    Cinnelli et al.

    Parameters:
    ----------
    results: list containing the SIR results as tuples of format: (seed_leaning, avg_leaning, len(rec)))
    """

    # define the number of bins and the number of values in each bin
    number_of_bins = round((np.sqrt(len(results)) / 3))
    values_per_bin = int(len(results) / number_of_bins)

    # create a list "nest" that contains a sublist with binned, consecutive values
    nest = [sorted(results)[i:i+values_per_bin] for i in range(0, number_of_bins*values_per_bin, values_per_bin)]

    # these lists for plotting contain: avg of polarity of the seed node, avg polarity of the influence set,
    # standard error for avg influence set polarity, and avg size for the values in each bin
    plot_list = []

    for i, _ in enumerate(nest):
        seed_mean = mean([ele[0] for ele in nest[i]])
        influence_mean = mean([ele[1] for ele in nest[i]])
        influence_sem = sem([ele[1] for ele in nest[i]])
        influence_size = mean([ele[2] for ele in nest[i]])

        plot_list.append((seed_mean, influence_mean, influence_sem, influence_size))

    return plot_list
