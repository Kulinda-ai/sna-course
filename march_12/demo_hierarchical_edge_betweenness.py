# FILE: demo_hierarchical_edge_betweenness.py

import json
import networkx as nx
import matplotlib.pyplot as plt

# Step 1: Load data and create the graph (same as before)
def create_graph_from_json(nodes_file_path, edges_file_path):
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

# Paths to your JSON files (update as needed)
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Step 2: Create the graph
G = create_graph_from_json(nodes_file_path, edges_file_path)

# Step 3: Compute edge betweenness centrality
edge_betweenness = nx.edge_betweenness_centrality(G, normalized=True)

# Step 4: Print edge betweenness centrality, sorted from highest to lowest
print("\n=== Edge Betweenness Centrality ===")
sorted_edges = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)

for idx, (edge, betweenness) in enumerate(sorted_edges):
    print(f"{idx + 1}. Edge {edge} -> Betweenness: {betweenness:.4f}")

# Optional: Draw the graph with edge thickness representing betweenness
def draw_graph_with_edge_betweenness(G, edge_betweenness):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 7))

    # Normalize widths for visibility
    edge_widths = [5 * edge_betweenness[edge] for edge in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray')
    nx.draw_networkx_labels(G, pos)

    plt.title("Graph with Edge Betweenness Highlighted (Edge Width ∝ Betweenness)")
    plt.axis('off')
    plt.show()

# Step 5: Visualize the graph with edge betweenness
draw_graph_with_edge_betweenness(G, edge_betweenness)
