# FILE: demo_hierarchical_edge_betweenness.py

import json
import networkx as nx
import matplotlib.pyplot as plt

# ==============================================================================
# STEP 1: Load Graph from JSON
# ==============================================================================

def create_graph_from_json(nodes_file_path, edges_file_path):
    """
    Loads nodes and edges from JSON files into an undirected graph.
    Each node may optionally include a 'label'.
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

# File paths
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Load the graph
G = create_graph_from_json(nodes_file_path, edges_file_path)

# ==============================================================================
# STEP 2: Compute Edge Betweenness Centrality
# ==============================================================================

# Edge betweenness centrality measures how often an edge lies on the shortest paths
# between nodes. High values mean the edge is a key bridge between regions of the graph.
edge_betweenness = nx.edge_betweenness_centrality(G, normalized=True)

# ==============================================================================
# STEP 3: Print Edges Sorted by Betweenness
# ==============================================================================

print("\n=== Edge Betweenness Centrality ===")
sorted_edges = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)

for idx, (edge, betweenness) in enumerate(sorted_edges):
    print(f"{idx + 1}. Edge {edge} -> Betweenness: {betweenness:.4f}")

# ==============================================================================
# STEP 4: Visualization (Edge Width ∝ Betweenness)
# ==============================================================================

def draw_graph_with_edge_betweenness(G, edge_betweenness):
    """
    Visualizes the graph and scales edge width by edge betweenness centrality.
    This makes the most 'important' edges visually stand out.
    """
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 7))

    # Scale edge widths: multiply betweenness by a constant to make it visible
    edge_widths = [5 * edge_betweenness[edge] for edge in G.edges()]

    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray')
    nx.draw_networkx_labels(G, pos)

    plt.title("Graph with Edge Betweenness Highlighted (Edge Width ∝ Betweenness)")
    plt.axis('off')
    plt.show()

# Run the visualizer
draw_graph_with_edge_betweenness(G, edge_betweenness)
