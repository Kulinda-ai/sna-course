# FILE: example_degree_centrality.py

import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# ==============================================================================
# STEP 1: Load graph from JSON files and build a NetworkX graph
# ==============================================================================

def create_graph_from_json(nodes_file_path, edges_file_path):
    # Load nodes
    with open(nodes_file_path, 'r') as file:
        nodes_data = json.load(file)

    # Load edges
    with open(edges_file_path, 'r') as file:
        edges_data = json.load(file)

    G = nx.Graph()  # Use undirected graph for general degree analysis

    # Add nodes with optional labels
    for node in nodes_data:
        G.add_node(node['id'], label=node.get('label', ''))

    # Add edges (assumes each edge connects a source and target node)
    for edge in edges_data:
        G.add_edge(edge['source'], edge['target'])

    return G

# File paths (adjust if needed)
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Build the graph object
G = create_graph_from_json(nodes_file_path, edges_file_path)

# ==============================================================================
# STEP 2: Calculate Degree Centrality
# ==============================================================================

# Degree centrality measures how many direct connections (edges) each node has.
# It’s normalized by default: max score = 1 if the node is connected to all others.
degree_centrality = nx.degree_centrality(G)

# Print each node's score to the console
print("\n=== Degree Centrality Scores ===")
for node, centrality in degree_centrality.items():
    print(f"{node}: {centrality:.4f}")

# ==============================================================================
# STEP 3: Rank Nodes by Degree Centrality
# ==============================================================================

# Convert the centrality dictionary into a ranked DataFrame
df = pd.DataFrame(list(degree_centrality.items()), columns=['Node', 'Degree Centrality'])
df = df.sort_values(by='Degree Centrality', ascending=False).reset_index(drop=True)
df.index += 1  # Set index to start from 1 for ranking

# Print the table to console
print("\n=== Ranked Nodes by Degree Centrality ===")
print(df)

# ==============================================================================
# STEP 4: Export Results
# ==============================================================================

# Save results to JSON file (one object per node)
df.to_json("degree_centrality.json", orient="records", indent=4)
print("\n✅ Saved degree centrality rankings to 'degree_centrality.json'.")
