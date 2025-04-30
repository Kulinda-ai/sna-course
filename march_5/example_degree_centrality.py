# FILE: example_degree_centrality.py

import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Create an empty graph
G = nx.Graph()

# Add nodes
def create_graph_from_json(nodes_file_path, edges_file_path):
    # Load nodes
    with open(nodes_file_path, 'r') as file:
        nodes_data = json.load(file)
    
    # Load edges
    with open(edges_file_path, 'r') as file:
        edges_data = json.load(file)
    
    # Create a new graph
    G = nx.Graph()
    
    # Add nodes to the graph
    for node in nodes_data:
        G.add_node(node['id'], label=node.get('label', ''))
    
    # Add edges to the graph
    for edge in edges_data:
        G.add_edge(edge['source'], edge['target'])
    
    return G

# Replace 'networkx_nodes.json' and 'networkx_edges.json' with the actual paths to your files
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Create the graph
G = create_graph_from_json(nodes_file_path, edges_file_path)

# Calculate degree centrality
degree_centrality = nx.degree_centrality(G)

# Print degree centrality of each node
for node, centrality in degree_centrality.items():
    print(f"{node}: {centrality}")

# Convert to DataFrame and Rank
df = pd.DataFrame(list(degree_centrality.items()), columns=['Node', 'Degree Centrality'])
df = df.sort_values(by='Degree Centrality', ascending=False).reset_index(drop=True)
df.index += 1  # Set ranking to start from 1

# Print DataFrame
print(df)

# Save to JSON file
df.to_json("degree_centrality.json", orient="records", indent=4)

print("\nSaved degree centrality rankings to 'degree_centrality.json'.")