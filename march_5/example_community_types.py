# FILE: example_community_types.py

import json
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import (
    greedy_modularity_communities,
    label_propagation_communities,
    girvan_newman
)

# ==============================================================================
# STEP 1: Load a graph from node and edge JSON files
# ==============================================================================

def create_graph_from_json(nodes_file_path, edges_file_path):
    with open(nodes_file_path, 'r') as file:
        nodes_data = json.load(file)
    
    with open(edges_file_path, 'r') as file:
        edges_data = json.load(file)
    
    G = nx.Graph()  # Use undirected graph for community detection

    for node in nodes_data:
        G.add_node(node['id'], label=node.get('label', ''))

    for edge in edges_data:
        G.add_edge(edge['source'], edge['target'])

    return G

# File paths to your dataset
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Load the graph
G = create_graph_from_json(nodes_file_path, edges_file_path)

# ==============================================================================
# STEP 2: Visualize Detected Communities
# ==============================================================================

def draw_communities(G, communities, title):
    pos = nx.spring_layout(G)  # Layout for visualization
    plt.figure(figsize=(10, 7))

    # Map each node to its community index (for coloring)
    node_color_map = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_color_map[node] = idx

    colors = [node_color_map.get(node, 0) for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.viridis, node_size=500)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos)

    plt.title(title)
    plt.axis('off')
    plt.show()

# ==============================================================================
# STEP 3: Greedy Modularity Communities
# ==============================================================================

# Greedy modularity tries to group nodes into communities to maximize modularity.
# It’s fast and often produces decent results for large undirected graphs.
greedy_communities = list(greedy_modularity_communities(G))
print(f"Greedy Modularity detected {len(greedy_communities)} communities.")
draw_communities(G, greedy_communities, "Greedy Modularity Communities")

# ==============================================================================
# STEP 4: Label Propagation Communities
# ==============================================================================

# Label Propagation is a fast, randomized method. 
# It spreads labels across the network until they stabilize into communities.
label_prop_communities = list(label_propagation_communities(G))
print(f"Label Propagation detected {len(label_prop_communities)} communities.")
draw_communities(G, label_prop_communities, "Label Propagation Communities")

# ==============================================================================
# STEP 5: Girvan-Newman Hierarchical Splits
# ==============================================================================

# Girvan-Newman removes edges with the highest betweenness centrality,
# gradually splitting the network into communities.
# It produces a hierarchy of nested communities.
girvan_newman_generator = girvan_newman(G)

# First split (typically produces 2 main communities)
first_level_communities = next(girvan_newman_generator)
print(f"Girvan-Newman (first split) detected {len(first_level_communities)} communities.")
draw_communities(G, first_level_communities, "Girvan-Newman Communities (First Split)")

# Optional: Second split to show deeper hierarchical layers
try:
    second_level_communities = next(girvan_newman_generator)
    print(f"Girvan-Newman (second split) detected {len(second_level_communities)} communities.")
    draw_communities(G, second_level_communities, "Girvan-Newman Communities (Second Split)")
except StopIteration:
    print("No more splits available.")
