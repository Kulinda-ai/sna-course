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

def predict_connections_for_cliques(graph):
    predictions = []
    for node in graph.nodes():
        neighbors = set(nx.neighbors(graph, node))
        for non_neighbor in set(graph.nodes()) - neighbors - {node}:
            common_neighbors = set(nx.common_neighbors(graph, node, non_neighbor))
            # We only consider pairs with at least two common neighbors to form a potential 3-node clique
            if len(common_neighbors) >= 2:
                prediction = {
                    "node": node,
                    "connected_node": non_neighbor,
                    'common_neighbors_count': len(common_neighbors),
                    "common_neighbors": list(common_neighbors)
                }
                predictions.append(prediction)
    return predictions

# Use the updated function
potential_connections_for_cliques = predict_connections_for_cliques(G)

# Convert to JSON string
connections_for_cliques_json = json.dumps(potential_connections_for_cliques, indent=4)

# Optionally, save to a file
with open('predicted_current_connections.json', 'w') as file:
    file.write(connections_for_cliques_json)

# Print part of the JSON string for demonstration
print(connections_for_cliques_json[:500])