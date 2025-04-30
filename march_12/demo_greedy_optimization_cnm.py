# FILE: demo_greedy_optimization_cnm.py

import json
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import greedy_modularity_communities

# ==============================================================================
# STEP 1: Load the Graph from JSON Files
# ==============================================================================

def create_graph_from_json(nodes_file_path, edges_file_path):
    """
    Load nodes and edges from JSON into a NetworkX graph (undirected).
    Each node is added with an optional 'label' field.
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

# ==============================================================================
# STEP 2: Visualize the Community Structure
# ==============================================================================

def draw_communities(G, communities, title):
    """
    Draw the graph with nodes colored by community membership.
    Each community is shown in a different color.
    """
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 7))

    # Assign community color to each node
    node_color_map = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_color_map[node] = idx

    colors = [node_color_map[node] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.viridis,
                           node_size=500, edgecolors='black')
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos)

    plt.title(title)
    plt.axis('off')
    plt.show()

# ==============================================================================
# STEP 3: Run CNM Greedy Modularity Optimization
# ==============================================================================

# Set file paths
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Load the graph
G = create_graph_from_json(nodes_file_path, edges_file_path)

# Run NetworkX’s implementation of Clauset-Newman-Moore (greedy modularity)
# This tries to find partitions that maximize modularity by greedily merging groups.
communities = list(greedy_modularity_communities(G))

# Output community stats
print(f"\n=== Greedy Modularity Communities (Clauset-Newman-Moore Algorithm) ===")
print(f"Total Communities Detected: {len(communities)}")
for idx, community in enumerate(communities, 1):
    print(f"Community {idx} ({len(community)} nodes): {sorted(community)}")

# ==============================================================================
# STEP 4: Compute Modularity Score
# ==============================================================================

# Modularity quantifies the quality of the partitioning (higher = better separation).
modularity = nx.algorithms.community.quality.modularity(G, communities)
print(f"\nModularity of partition: {modularity:.4f}")

# ==============================================================================
# STEP 5: Visualize Communities
# ==============================================================================

draw_communities(G, communities, "Greedy Modularity Communities (CNM Algorithm)")
