# FILE: demo_hierarchical_community_dendrogram.py

import json
import networkx as nx
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import numpy as np

# ==============================================================================
# STEP 1: Load Data and Create Graph
# ==============================================================================

def create_graph_from_json(nodes_file_path, edges_file_path):
    """
    Loads a graph from node and edge JSON files.
    Each node must have an 'id'; optional labels can be included.
    """
    with open(nodes_file_path, 'r') as file:
        nodes_data = json.load(file)
    with open(edges_file_path, 'r') as file:
        edges_data = json.load(file)

    G = nx.Graph()

    for node in nodes_data:
        G.add_node(node['id'], label=node.get('label', ''))

    for edge in edges_data:
        G.add_edge(edge['source'], edge['target'])

    return G

# Set file paths
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Create graph
G = create_graph_from_json(nodes_file_path, edges_file_path)

# ==============================================================================
# STEP 2: Compute the Distance Matrix
# ==============================================================================

# Create adjacency matrix (1 if edge exists, 0 otherwise)
# This assumes a simple binary unweighted graph
adj_matrix = nx.to_numpy_array(G)

# Convert adjacency to a distance matrix:
# - Distance = 0 if connected
# - Distance = 1 if not connected
distance_matrix = 1 - adj_matrix

# Set diagonal to 0 (distance from node to itself)
np.fill_diagonal(distance_matrix, 0)

# Sanity check
assert np.allclose(np.diag(distance_matrix), 0), "Diagonal should be all zeros"

# ==============================================================================
# STEP 3: Convert Matrix to Condensed Format for Clustering
# ==============================================================================

# Convert full square distance matrix into a condensed 1D array
# This is the format expected by `scipy.linkage`
condensed_distance = squareform(distance_matrix)

# ==============================================================================
# STEP 4: Compute Linkage for Agglomerative Clustering
# ==============================================================================

# Perform hierarchical/agglomerative clustering using the linkage method
# Options for `method` include 'single', 'complete', 'average'
Z = linkage(condensed_distance, method='average')

# `Z` encodes the merge order and distances — used to build the dendrogram

# ==============================================================================
# STEP 5: Plot the Dendrogram
# ==============================================================================

plt.figure(figsize=(12, 8))

# Visualize the hierarchical clustering
# Each merge step is shown as a horizontal line
dendrogram(Z, labels=[str(n) for n in G.nodes()])

plt.title("Hierarchical Dendrogram (Agglomerative Clustering on Your Dataset)")
plt.xlabel("Nodes")
plt.ylabel("Distance (Edge Disconnection)")
plt.show()
