# FILE: demo_hierarchical_community_5.py

import json
import networkx as nx
import matplotlib.pyplot as plt

# ==============================================================================
# STEP 1: Load Graph from JSON
# ==============================================================================

def create_graph_from_json(nodes_file_path, edges_file_path):
    """
    Loads nodes and edges from JSON files and builds an undirected graph.
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

# Set paths to input files
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Load graph from files
G = create_graph_from_json(nodes_file_path, edges_file_path)

# ==============================================================================
# STEP 2: Draw Graph with Nodes Colored by Community
# ==============================================================================

def draw_communities_step(G, communities, step):
    """
    Draws the network with each node colored by its community index.
    Used to visualize the community structure at each iteration.
    """
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 7))

    # Assign a color index to each node
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
# STEP 3: Step-by-Step Girvan-Newman Algorithm
# ==============================================================================

def girvan_newman_step_by_step(G, max_steps=10):
    """
    Runs Girvan-Newman community detection iteratively, removing edges
    with the highest betweenness centrality at each step and visualizing the result.
    """
    G_copy = G.copy()  # Work on a copy to preserve original
    step = 0

    # Initial community: entire graph is one big component
    components = list(nx.connected_components(G_copy))
    print(f"Step {step}: {len(components)} communities")
    for idx, community in enumerate(components):
        print(f"  Community {idx + 1}: {sorted(community)}")

    draw_communities_step(G_copy, components, step)

    while step < max_steps:
        step += 1

        # Compute edge betweenness centrality (importance of edges)
        edge_betweenness = nx.edge_betweenness_centrality(G_copy)

        if not edge_betweenness:
            print("No more edges to remove. Stopping.")
            break

        # Find edge(s) with the highest betweenness
        max_bw = max(edge_betweenness.values())
        edges_to_remove = [edge for edge, bw in edge_betweenness.items() if bw == max_bw]

        print(f"\nStep {step}: Removing {len(edges_to_remove)} edge(s) with highest betweenness ({max_bw:.4f})")
        for edge in edges_to_remove:
            print(f"  Removing edge: {edge}")

        # Remove the most "central" edges
        G_copy.remove_edges_from(edges_to_remove)

        # Identify resulting communities (connected components)
        components = list(nx.connected_components(G_copy))
        print(f"After removal: {len(components)} communities")
        for idx, community in enumerate(components):
            print(f"  Community {idx + 1}: {sorted(community)}")

        # Visualize current partition
        draw_communities_step(G_copy, components, step)

        # Optional: break early if desired
        # if len(components) >= 5:
        #     break

    print("\nFinal communities after step-by-step Girvan-Newman:")
    for idx, community in enumerate(components):
        print(f"  Community {idx + 1}: {sorted(community)}")

    return components

# Run the community detection with a limited number of steps
final_components = girvan_newman_step_by_step(G, max_steps=5)
