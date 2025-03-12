import json
import networkx as nx
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import numpy as np

# Step 1: Load data and create the graph (as before)
def create_graph_from_json(nodes_file_path, edges_file_path):
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

# Paths to your JSON files
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Step 2: Create the graph from your data
G = create_graph_from_json(nodes_file_path, edges_file_path)

# Step 3: Compute the distance matrix
# Create adjacency matrix (binary: 1 if edge exists, 0 otherwise)
adj_matrix = nx.to_numpy_array(G)

# Invert adjacency to create a "distance" matrix
# Where connected nodes are 0 distance, unconnected are 1 distance
distance_matrix = 1 - adj_matrix

# Step 4: Fix the diagonal (self-distance should be 0)
np.fill_diagonal(distance_matrix, 0)

# Confirm the diagonal is zero
assert np.allclose(np.diag(distance_matrix), 0), "Diagonal is not zero!"

# Step 5: Convert to condensed distance format (needed for linkage)
condensed_distance = squareform(distance_matrix)

# Step 6: Compute linkage matrix using agglomerative clustering
Z = linkage(condensed_distance, method='average')  # or 'single', 'complete'

# Step 7: Plot dendrogram
plt.figure(figsize=(12, 8))
dendrogram(Z, labels=[str(n) for n in G.nodes()])
plt.title("Hierarchical Dendrogram (Agglomerative Clustering on Your Dataset)")
plt.xlabel("Nodes")
plt.ylabel("Distance (Edge Disconnection)")
plt.show()
