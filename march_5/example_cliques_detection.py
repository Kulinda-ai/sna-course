# FILE: example_cliques_detection.py

import json
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain  # This import is unused in this script but often useful in related community work
import pandas as pd

# ==============================================================================
# STEP 1: Load nodes and edges from JSON and build an undirected graph
# ==============================================================================

def create_graph_from_json(nodes_file_path, edges_file_path):
    # Load node and edge data from disk
    with open(nodes_file_path, 'r') as file:
        nodes_data = json.load(file)
    
    with open(edges_file_path, 'r') as file:
        edges_data = json.load(file)
    
    G = nx.Graph()  # Using an undirected graph because cliques are defined over undirected edges

    # Add nodes (with optional labels)
    for node in nodes_data:
        G.add_node(node['id'], label=node.get('label', ''))
    
    # Add edges between nodes
    for edge in edges_data:
        G.add_edge(edge['source'], edge['target'])
    
    return G

# Paths to your input files
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Build the graph from data
G = create_graph_from_json(nodes_file_path, edges_file_path)

# ==============================================================================
# STEP 2: Detect Maximal Cliques
# ==============================================================================

# A clique is a group of nodes where every node is connected to every other node.
# A **maximal clique** is a clique that cannot be extended by including any adjacent node.
# We're interested in finding tight-knit groups (e.g. 3+ people who all know each other).

cliques = list(nx.find_cliques(G))  # Uses the Bron–Kerbosch algorithm under the hood

# Filter for cliques with 3 or more nodes (triangles and larger)
filtered_cliques = [clique for clique in cliques if len(clique) >= 3]

# ==============================================================================
# STEP 3: Format Results as JSON
# ==============================================================================

# Build a JSON-style list of dictionaries, each representing a clique
filtered_cliques_json_array = [
    {"clique_id": i, "nodes": clique}
    for i, clique in enumerate(filtered_cliques)
]

# Convert to nicely formatted JSON string
filtered_cliques_json = json.dumps(filtered_cliques_json_array, indent=4)

# Save to file
with open('cliques.json', 'w') as file:
    file.write(filtered_cliques_json)

# Print first 500 characters to console as a preview
print(filtered_cliques_json[:500])
