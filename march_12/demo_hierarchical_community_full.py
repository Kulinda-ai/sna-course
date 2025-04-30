# FILE: demo_hierarchical_community_full.py

import json
import networkx as nx
import matplotlib.pyplot as plt

# ==============================================================================
# STEP 1: Load Graph from JSON Files
# ==============================================================================

def create_graph_from_json(nodes_file_path, edges_file_path):
    """
    Loads node and edge data from JSON files and builds a NetworkX undirected graph.
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

# File paths for node/edge files
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Load the graph
G = create_graph_from_json(nodes_file_path, edges_file_path)

# ==============================================================================
# STEP 2: Visualization Function for Final Community View
# ==============================================================================

def draw_communities_final(G, communities, title):
    """
    Draws a graph layout with nodes colored by their final community assignment.
    Each node gets its own color at the end (1 node = 1 community).
    """
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 7))

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

# ==============================================================================
# STEP 3: Full Girvan-Newman Decomposition
# ==============================================================================

def girvan_newman_until_isolated(G):
    """
    Fully applies the Girvan-Newman algorithm by removing the highest betweenness
    edges one step at a time until the graph is fully disconnected (1 node per community).
    """
    G_copy = G.copy()
    step = 0

    while G_copy.number_of_edges() > 0:
        step += 1

        # Compute betweenness centrality of edges
        edge_betweenness = nx.edge_betweenness_centrality(G_copy)

        if not edge_betweenness:
            print("No more edges to remove.")
            break

        # Identify edges with the highest betweenness
        max_bw = max(edge_betweenness.values())
        edges_to_remove = [edge for edge, bw in edge_betweenness.items() if bw == max_bw]

        print(f"Step {step}: Removing {len(edges_to_remove)} edge(s) with betweenness {max_bw:.4f}")
        G_copy.remove_edges_from(edges_to_remove)

    # Final communities = disconnected nodes (each is its own component)
    final_components = list(nx.connected_components(G_copy))

    print("\nFinal communities (each node is its own component):")
    for idx, community in enumerate(final_components):
        print(f"  Community {idx + 1}: {sorted(community)}")

    # Visualize isolated nodes
    draw_communities_final(G_copy, final_components, "Final: All Nodes Isolated")

    return final_components

# Run the algorithm until all nodes are disconnected
final_components = girvan_newman_until_isolated(G)
