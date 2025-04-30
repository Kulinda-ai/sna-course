# FILE: example_eigenvector_centrality.py

import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# ==============================================================================
# STEP 1: Load Graph from JSON (Nodes and Edges)
# ==============================================================================

def create_graph_from_json(nodes_file_path, edges_file_path):
    """
    Reads node and edge data from JSON files and creates an undirected NetworkX graph.
    Each node may contain an optional 'label'.
    """
    # Load node data
    with open(nodes_file_path, 'r') as file:
        nodes_data = json.load(file)

    # Load edge data
    with open(edges_file_path, 'r') as file:
        edges_data = json.load(file)

    # Create an undirected graph
    G = nx.Graph()

    # Add nodes with optional label
    for node in nodes_data:
        G.add_node(node['id'], label=node.get('label', ''))

    # Add edges
    for edge in edges_data:
        G.add_edge(edge['source'], edge['target'])

    return G

# Load the network from files
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'
G = create_graph_from_json(nodes_file_path, edges_file_path)

# ==============================================================================
# STEP 2: Compute Eigenvector Centrality
# ==============================================================================

# Eigenvector centrality measures not just how many connections a node has,
# but how important those connections are — it gives higher weight to nodes
# connected to other central nodes.
eigenvector_centrality = nx.eigenvector_centrality(G)

# Print out the eigenvector centrality score for each node
for node, centrality in eigenvector_centrality.items():
    print(f"{node}: {centrality:.6f}")

# ==============================================================================
# STEP 3: Store, Rank, and Export Results
# ==============================================================================

# Convert to DataFrame for sorting and inspection
df = pd.DataFrame(list(eigenvector_centrality.items()), columns=['Node', 'Eigenvector Centrality'])
df = df.sort_values(by='Eigenvector Centrality', ascending=False).reset_index(drop=True)
df.index += 1  # Start index at 1 for ranking

# Print ranked table
print("\n=== Eigenvector Centrality Rankings ===")
print(df)

# Save ranked results to JSON
df.to_json("eigenvector_centrality.json", orient="records", indent=4)
print("\n✅ Saved eigenvector centrality rankings to 'eigenvector_centrality.json'.")
