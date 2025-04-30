# FILE: example_community_types.py

import json
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import (
    greedy_modularity_communities,
    label_propagation_communities,
    girvan_newman
)

# Load data and create the graph
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

# Replace with your actual file paths
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Create graph
G = create_graph_from_json(nodes_file_path, edges_file_path)

# Helper function to visualize communities
def draw_communities(G, communities, title):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 7))
    
    # Flatten communities into a node -> community index mapping
    node_color_map = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_color_map[node] = idx

    colors = [node_color_map[node] for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.viridis, node_size=500)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos)
    plt.title(title)
    plt.axis('off')
    plt.show()

# 1. Greedy Modularity Communities
greedy_communities = list(greedy_modularity_communities(G))
print(f"Greedy Modularity detected {len(greedy_communities)} communities.")
draw_communities(G, greedy_communities, "Greedy Modularity Communities")

# 2. Label Propagation Communities
label_prop_communities = list(label_propagation_communities(G))
print(f"Label Propagation detected {len(label_prop_communities)} communities.")
draw_communities(G, label_prop_communities, "Label Propagation Communities")

# 3. Girvan-Newman Communities
# Get first level of communities (split into 2 groups)
girvan_newman_generator = girvan_newman(G)
first_level_communities = next(girvan_newman_generator)  # First split
print(f"Girvan-Newman (first split) detected {len(first_level_communities)} communities.")
draw_communities(G, first_level_communities, "Girvan-Newman Communities (First Split)")

# Optional: Get next split if you want to go deeper
try:
    second_level_communities = next(girvan_newman_generator)
    print(f"Girvan-Newman (second split) detected {len(second_level_communities)} communities.")
    draw_communities(G, second_level_communities, "Girvan-Newman Communities (Second Split)")
except StopIteration:
    print("No more splits available.")