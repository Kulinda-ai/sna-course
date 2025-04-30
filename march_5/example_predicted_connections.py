# FILE: example_predicted_connections.py

import json
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain  # not used in this script but often used for community detection
import pandas as pd

# ==============================================================================
# STEP 1: Load Graph from JSON Files
# ==============================================================================

def create_graph_from_json(nodes_file_path, edges_file_path):
    """
    Loads a graph from node and edge JSON files. Each node has an ID and optional label.
    The graph is created as undirected for simplicity in link prediction.
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

# Build the graph
G = create_graph_from_json(nodes_file_path, edges_file_path)

# ==============================================================================
# STEP 2: Predict Possible Connections Based on Shared Neighbors
# ==============================================================================

def predict_connections_for_cliques(graph):
    """
    Predict potential connections using clique-based logic.
    If two nodes share 2 or more neighbors, they are likely to form a 3-node clique (triangle).
    This is a simplified heuristic for link prediction.
    """
    predictions = []

    for node in graph.nodes():
        neighbors = set(nx.neighbors(graph, node))

        # Compare to all non-neighbors (excluding itself)
        for non_neighbor in set(graph.nodes()) - neighbors - {node}:
            common_neighbors = set(nx.common_neighbors(graph, node, non_neighbor))

            # Only predict if there are at least 2 shared neighbors (potential triangle)
            if len(common_neighbors) >= 2:
                prediction = {
                    "node": node,
                    "connected_node": non_neighbor,
                    "common_neighbors_count": len(common_neighbors),
                    "common_neighbors": list(common_neighbors)
                }
                predictions.append(prediction)

    return predictions

# Run the prediction function
potential_connections_for_cliques = predict_connections_for_cliques(G)

# ==============================================================================
# STEP 3: Output Predicted Connections
# ==============================================================================

# Convert to formatted JSON for viewing/saving
connections_for_cliques_json = json.dumps(potential_connections_for_cliques, indent=4)

# Save to file
with open('predicted_current_connections.json', 'w') as file:
    file.write(connections_for_cliques_json)

# Print sample to console
print(connections_for_cliques_json[:500])
