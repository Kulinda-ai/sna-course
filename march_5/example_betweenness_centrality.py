# FILE: example_betweenness_centrality.py

import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------------------------------------------------------------
# STEP 1: Load nodes and edges from JSON and build an undirected graph
# ------------------------------------------------------------------------------

def create_graph_from_json(nodes_file_path, edges_file_path):
    with open(nodes_file_path, 'r') as file:
        nodes_data = json.load(file)
    
    with open(edges_file_path, 'r') as file:
        edges_data = json.load(file)
    
    G = nx.Graph()  # We use an undirected graph for general centrality analysis
    
    for node in nodes_data:
        G.add_node(node['id'], label=node.get('label', ''))  # Optionally preserve labels
    
    for edge in edges_data:
        G.add_edge(edge['source'], edge['target'])  # Assumes edges are unweighted

    return G

# Provide file paths to your networkx-style node and edge files
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Build the graph
G = create_graph_from_json(nodes_file_path, edges_file_path)

# ------------------------------------------------------------------------------
# STEP 2: Compute Betweenness Centrality
# ------------------------------------------------------------------------------

# Betweenness centrality measures how often a node lies on shortest paths between other nodes.
# High betweenness = the node is a bridge or bottleneck in the network.
# NetworkX computes this using Brandes' algorithm (efficient for large graphs).
betweenness_centrality = nx.betweenness_centrality(G, normalized=True)

# ------------------------------------------------------------------------------
# STEP 3: Print betweenness scores to the console
# ------------------------------------------------------------------------------

print("\n=== Betweenness Centrality (Raw Scores) ===")
for node, centrality in betweenness_centrality.items():
    print(f"{node}: {centrality:.4f}")

# ------------------------------------------------------------------------------
# STEP 4: Convert results to a ranked pandas DataFrame
# ------------------------------------------------------------------------------

df = pd.DataFrame(
    list(betweenness_centrality.items()),
    columns=['Node', 'Betweenness Centrality']
)

# Rank nodes by their betweenness score (descending)
df = df.sort_values(by='Betweenness Centrality', ascending=False).reset_index(drop=True)
df.index += 1  # Start ranking at 1 (more intuitive)

# ------------------------------------------------------------------------------
# STEP 5: Display the ranked centrality table
# ------------------------------------------------------------------------------

print("\n=== Ranked Nodes by Betweenness Centrality ===")
print(df)

# ------------------------------------------------------------------------------
# STEP 6: Save centrality rankings to a JSON file
# ------------------------------------------------------------------------------

df.to_json("betweenness_centrality.json", orient="records", indent=4)
print("\n✅ Saved betweenness centrality rankings to 'betweenness_centrality.json'.")
