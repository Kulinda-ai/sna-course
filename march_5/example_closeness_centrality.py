# FILE: example_closeness_centrality.py

import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# ==============================================================================
# STEP 1: Load Graph from JSON Files
# ==============================================================================

def create_graph_from_json(nodes_file_path, edges_file_path):
    # Load node list
    with open(nodes_file_path, 'r') as file:
        nodes_data = json.load(file)
    
    # Load edge list
    with open(edges_file_path, 'r') as file:
        edges_data = json.load(file)
    
    G = nx.Graph()  # Use an undirected graph for centrality analysis

    # Add nodes to graph with optional labels
    for node in nodes_data:
        G.add_node(node['id'], label=node.get('label', ''))
    
    # Add edges between nodes
    for edge in edges_data:
        G.add_edge(edge['source'], edge['target'])
    
    return G

# File paths (adjust if needed)
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Create graph object
G = create_graph_from_json(nodes_file_path, edges_file_path)

# ==============================================================================
# STEP 2: Calculate Closeness Centrality
# ==============================================================================

# Closeness centrality measures how close a node is to all others in the network.
# Nodes with high closeness can quickly interact or reach many other nodes.
# It is computed as the reciprocal of the average shortest path length from the node to all others.
closeness_centrality = nx.closeness_centrality(G)

# Print raw scores to console
print("\n=== Closeness Centrality Scores ===")
for node, centrality in closeness_centrality.items():
    print(f"{node}: {centrality:.4f}")

# ==============================================================================
# STEP 3: Convert Results to DataFrame for Ranking
# ==============================================================================

df = pd.DataFrame(
    list(closeness_centrality.items()),
    columns=['Node', 'Closeness Centrality']
)

# Sort from highest to lowest (most central = rank 1)
df = df.sort_values(by='Closeness Centrality', ascending=False).reset_index(drop=True)
df.index += 1  # Start ranking from 1

# Print ranked table
print("\n=== Ranked Nodes by Closeness Centrality ===")
print(df)

# ==============================================================================
# STEP 4: Export Results to JSON File
# ==============================================================================

df.to_json("closeness_centrality.json", orient="records", indent=4)
print("\n✅ Saved closeness centrality rankings to 'closeness_centrality.json'.")
