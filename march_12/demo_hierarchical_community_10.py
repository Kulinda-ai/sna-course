# FILE: demo_hierarchical_community_10.py

import json
import networkx as nx
import matplotlib.pyplot as plt

# ==============================================================================
# STEP 1: Load Graph from JSON
# ==============================================================================

def create_graph_from_json(nodes_file_path, edges_file_path):
    """
    Loads nodes and edges from JSON files and constructs an undirected graph.
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

# Set paths to your node/edge files
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Create the graph
G = create_graph_from_json(nodes_file_path, edges_file_path)

# ==============================================================================
# STEP 2: Draw the Communities at Each Step
# ==============================================================================

def draw_communities_step(G, communities, step):
    """
    Visualizes the community partition at a given step.
    Each community is assigned a unique color.
    """
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 7))

    # Map each node to its community index
    node_color_map = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_color_map[node] = idx

    colors = [node_color_map.get(node, -1) for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.viridis, node_size=500)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos)

    plt.title(f"Step {step}: {len(communities)} Communities")
    plt.axis('off')
    plt.show()

# ==============================================================================
# STEP 3: Girvan-Newman Hierarchical Split (10 Steps)
# ==============================================================================

def girvan_newman_step_by_step(G, max_steps=10):
    """
    Perform a manual step-by-step Girvan-Newman algorithm to observe how communities split over time.
    At each step, edges with highest betweenness centrality are removed.
    """
    G_copy = G.copy()
    step = 0

    # Initially, the whole graph is one community
    components = list(nx.connected_components(G_copy))
    print(f"Step {step}: {len(components)} communities")
    for idx, community in enumerate(components):
        print(f"  Community {idx + 1}: {sorted(community)}")

    draw_communities_step(G_copy, components, step)

    while step < max_steps:
        step += 1

        # Calculate edge betweenness (importance as bridges)
        edge_betweenness = nx.edge_betweenness_centrality(G_copy)

        if not edge_betweenness:
            print("No more edges to remove. Stopping.")
            break

        # Identify and remove edges with highest betweenness
        max_bw = max(edge_betweenness.values())
        edges_to_remove = [edge for edge, bw in edge_betweenness.items() if bw == max_bw]

        print(f"\nStep {step}: Removing {len(edges_to_remove)} edge(s) with highest betweenness ({max_bw:.4f})")
        for edge in edges_to_remove:
            print(f"  Removing edge: {edge}")

        G_copy.remove_edges_from(edges_to_remove)

        # Recalculate components as new communities
        components = list(nx.connected_components(G_copy))
        print(f"After removal: {len(components)} communities")
        for idx, community in enumerate(components):
            print(f"  Community {idx + 1}: {sorted(community)}")

        draw_communities_step(G_copy, components, step)

        # Stop early if the graph has no more edges to split
        if G_copy.number_of_edges() == 0:
            print("No edges left in the graph. Stopping.")
            break

    print("\nFinal communities after step-by-step Girvan-Newman:")
    for idx, community in enumerate(components):
        print(f"  Community {idx + 1}: {sorted(community)}")

    return components

# Run the detection process for 10 iterations
final_components = girvan_newman_step_by_step(G, max_steps=10)
