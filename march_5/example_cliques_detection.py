# FILE: example_cliques_detection.py

import json
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
import pandas as pd

# Create an empty graph
G = nx.Graph()

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

# Find all maximal cliques in the graph
cliques = list(nx.find_cliques(G))

# Filter cliques to include only those with 3 or more nodes
filtered_cliques = [clique for clique in cliques if len(clique) >= 3]

# Convert filtered cliques into the specified JSON format
filtered_cliques_json_array = [
    {"clique_id": i, "nodes": clique}
    for i, clique in enumerate(filtered_cliques)
]

# Convert to JSON string
filtered_cliques_json = json.dumps(filtered_cliques_json_array, indent=4)

# Optionally, save to a file
with open('cliques.json', 'w') as file:
    file.write(filtered_cliques_json)

# Print the JSON string
print(filtered_cliques_json[:500])  