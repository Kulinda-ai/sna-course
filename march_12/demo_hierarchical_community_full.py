# FILE: demo_hierarchical_community_full.py

import json
import networkx as nx
import matplotlib.pyplot as plt

# Load data and create the graph (same as before)
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

# Paths to your JSON files
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Create the graph
G = create_graph_from_json(nodes_file_path, edges_file_path)

# Function to draw communities (optional final step)
def draw_communities_final(G, communities, title):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 7))

    # Flatten communities into node -> community index
    node_color_map = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_color_map[node] = idx

    colors = [node_color_map.get(node, -1) for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.viridis, node_size=500)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Girvan-Newman: Break the graph until every node is its own community
def girvan_newman_until_isolated(G):
    # Make a copy so we don't modify the original
    G_copy = G.copy()

    step = 0
    total_nodes = len(G_copy.nodes())

    while G_copy.number_of_edges() > 0:
        step += 1
        edge_betweenness = nx.edge_betweenness_centrality(G_copy)

        if not edge_betweenness:
            print("No more edges to remove.")
            break

        # Remove edges with highest betweenness
        max_bw = max(edge_betweenness.values())
        edges_to_remove = [edge for edge, bw in edge_betweenness.items() if bw == max_bw]

        print(f"Step {step}: Removing {len(edges_to_remove)} edge(s) with betweenness {max_bw:.4f}")
        G_copy.remove_edges_from(edges_to_remove)

    # After all edges are removed, get the final components (each node should be isolated)
    final_components = list(nx.connected_components(G_copy))

    print("\nFinal communities (each node is its own component):")
    for idx, community in enumerate(final_components):
        print(f"  Community {idx + 1}: {sorted(community)}")

    # Optional: Draw final isolated nodes
    draw_communities_final(G_copy, final_components, "Final: All Nodes Isolated")

    return final_components

# Run the full split to isolation
final_components = girvan_newman_until_isolated(G)
