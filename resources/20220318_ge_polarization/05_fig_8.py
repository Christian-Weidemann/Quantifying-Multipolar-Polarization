import sys
import pandas as pd
import networkx as nx
from modules import ps
from itertools import combinations
from scipy.stats import gaussian_kde

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)
pd.options.mode.chained_assignment = None

# This function creates all possible pairings of congressment in a single congress, together with the number they co-voted
# Each roll call is a vote and each vote is a 0/1 value depending on how the congressman voted
# By grouping by rollcall and vote we create a group of all the congressman casting the same vote at the same roll call (i.e. they voted the same thing in the same way)
# We then group by again, counting the number of times this happened
def make_all_edges(df):
    df = df.groupby(by = ["rollnumber", "vote"]).apply(lambda x: pd.DataFrame(list(combinations(x["icpsr"], 2)))) # "combinations" makes all possible pairs of icpsr codes for every vote value
    df.columns = ("src", "trg")
    df = df.groupby(by = ["src", "trg"]).size().reset_index().rename(columns = {0: "nij"})                        # Counts how many times a pair of congressmen appears in df (i.e. they co-voted)
    return df

# This function makes a KDE of the probabilities two congressmen co-voted.
# It makes one for pairs of congressmen in the samr party (sp_pdf) and for pairs in different parties (cp_pdf, cp = cross-party)
def make_pdfs(edges):
    edges = edges.merge(nodes[nodes["congress"] == congress], left_on = "src", right_on = "icpsr")                              # Merge co-voting counts with nodes to get party infor for the src node
    edges = edges.merge(nodes[nodes["congress"] == congress], left_on = "trg", right_on = "icpsr", suffixes = ("_src", "_trg")) # Same as above, for the trg node
    edges["nij"] /= edges["nij"].max()                                                                                          # Transforms co-vote counts into probabilities, normalized by max count
    edges["same_party"] = edges["party_code_src"] == edges["party_code_trg"]                                                    # Add a flag: True if nodes are in the same party, false otherwise
    sp_pdf = gaussian_kde(edges[edges["same_party"]]["nij"])                                                                    # Make KDE only with same party data
    cp_pdf = gaussian_kde(edges[~edges["same_party"]]["nij"])                                                                   # Make KDE only with cross party data
    return edges, sp_pdf,cp_pdf

# Given two PDFs (kde1, kde2), this function finds the first intersection they have after scope[0]
def find_intersection(kde1, kde2, init_interval = 0.01, scope = [0,1], convergence = 0.0001):
    x_left = scope[0]                                    # We start at the leftmost part of our scope
    x_right = scope[0] + init_interval
    while x_right < scope[1]:                            # Keep going until we're out of scope
        left = kde1(x_left)[0] - kde2(x_left)[0]
        right = kde1(x_right)[0] - kde2(x_right)[0]
        if left * right < 0:                             # If the functions intersected (an odd number of times) in the interval...
            if init_interval <= convergence:             # ...and the interval is small enough for us to be happy with it...
                return x_right                           # ...then we found the crossing point
            else:                                        # Otherwise narrow down the interval and search again
                return find_intersection(kde1, kde2, init_interval / 10, scope = [x_left, x_right])
        else:                                            # If no intersection or an even number of intersections in the interval...
            x_left = x_right                             # ...then shift the interval rightwards
            x_right += init_interval
    return scope[0]

# This function writes the nodes with their NOMINATE score and the edges weighted by the mean NOMINATE scores of the nodes they connect
# Only prints the edges with a value higher than the threshold calculated with the function above (the intersection of the sp and cp PDFs)
def print_network(edges, nodes, congress, threshold):
    edges = edges[edges["nij"] > threshold]
    edges[["src", "trg"]] = edges[["src", "trg"]].astype(int)
    edges["pol"] = edges[["nominate_dim1_src", "nominate_dim1_trg"]].mean(axis = 1)
    edges[["src", "trg", "nij", "pol"]].to_csv(f"congress{congress}_edges.csv", sep = "\t", index = False)
    nodes[nodes["congress"] == congress].to_csv(f"congress{congress}_nodes.csv", sep = "\t", index = False)

# Bin the NOMINATE values and count how many congressmen are inside each bin.
def make_hist(congress, nodes):
    bins = pd.IntervalIndex.from_breaks(list([-(20 - _) / 20 for _ in range(41)]))
    labels = [(b.left + b.right) / 2 for b in bins]
    df = nodes[nodes["congress"] == congress][["icpsr", "nominate_dim1"]]
    df.columns = ("user", "pol")
    df["pol"] = pd.cut(df["pol"], bins).cat.rename_categories(labels)
    df = df.groupby(by = "pol").size().reset_index()
    df.columns = ("pol", "count")
    df.to_csv(f"{congress}_nodes_hist.csv", index = False, sep = "\t")

# Make the G and o objects to calculate delta and return delta
def calc_pol(congress):
    G = nx.read_edgelist(f"congress{congress}_edges.csv", delimiter = '\t', comments = 's', data = False, nodetype = int)
    o = pd.read_csv(f"congress{congress}_nodes.csv", sep = "\t").set_index("icpsr")["nominate_dim1"].to_dict()
    return ps.ge(o, {}, G)

df = pd.read_csv("HSall_votes.csv")                                                                       # Load the vote data
nodes = pd.read_csv("HSall_members.csv")                                                                  # Load the NOMINATe scores
nodes = nodes[["icpsr", "congress", "party_code", "nominate_dim1"]]                                       # We only care about the first dimension of NOMINATE
df["vote"] = df["cast_code"].map(lambda x: 1 if x in (1, 2, 3) else 0 if x in (4, 5, 6) else -1)          # Every no type (1,2,3) is a no and every aye type (4,5,6) is an aye. We ignore abstentions
df = df[(df["vote"] != -1) & (df["chamber"] == "House")].drop(["chamber", "cast_code", "prob"], axis = 1) # Drop abstentions and consider only votes in the House. Drop useless data

df["icpsr"] = df["icpsr"].astype(int)                                 # Convert ICPSR to integer

for congress in range(81, 117):                                       # Loop over each Congress
   sys.stderr.write(f"Making congress {congress}\n")
   edges = make_all_edges(df[(df["congress"] == congress)])           # Make edges using exclusively data from the currently considered congress
   edges, sp_pdf, cp_pdf = make_pdfs(edges)                           # Create the probability density functions for "same party" and "cross party" events
   threshold = find_intersection(sp_pdf, cp_pdf, scope = [0.4, 1])    # Find where the two PDFs intersect in the first non-trivial point (which should be beyond 0.4)
   sys.stderr.write(f"Found threshold at {threshold}\n\n")
   print_network(edges, nodes, congress, threshold)                   # Print the network

for congress in (81, 88, 95, 102, 109, 116):
   make_hist(congress, nodes)                                         # Make the histograms counting how many congressment are in a specific NOMINATE bin

with open("congress_pol.csv", 'w') as f:
   for congress in range(81, 117):
      f.write(f"{congress}\t{calc_pol(congress)}\n")                  # Write the delta value of each congress to file
      f.flush()
