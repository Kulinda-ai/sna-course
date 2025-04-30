# FILE: demo_label_proposition.py

import json
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import random

# ==============================================================================
# STEP 1: Load Graph from JSON
# ==============================================================================

def create_graph_from_json(nodes_file_path, edges_file_path):
    """
    Loads nodes and edges from JSON and builds an undirected graph.
    Each node is assigned an optional label.
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
# STEP 2: Manual Label Propagation Algorithm
# ==============================================================================

def manual_label_propagation(G, max_iterations=10):
    """
    Custom implementation of the Label Propagation Algorithm.
    Nodes adopt the most frequent label among their neighbors.
    The process repeats until labels stabilize or max_iterations is reached.
    """
    # Initialize: Each node starts with a unique label (its own ID)
    labels = {node: node for node in G.nodes()}

    print("Initial labels:")
    for node, label in labels.items():
        print(f"  Node {node}: Label {label}")
    print("\n")

    # Iteratively update labels
    for iteration in range(max_iterations):
        print(f"=== Iteration {iteration + 1} ===")
        nodes = list(G.nodes())
        random.shuffle(nodes)  # Shuffle to simulate asynchronous updates

        changes = 0  # Track how many labels changed in this round

        for node in nodes:
            neighbor_labels = [labels[neighbor] for neighbor in G.neighbors(node)]
            if not neighbor_labels:
                continue  # Skip isolated nodes

            # Count frequency of neighbor labels
            label_count = Counter(neighbor_labels)
            max_count = max(label_count.values())
            most_common_labels = [label for label, count in label_count.items() if count == max_count]

            # Tie-breaking randomly if multiple labels are equally frequent
            new_label = random.choice(most_common_labels)

            # Update label if different from current
            if labels[node] != new_label:
                print(f"  Node {node} changed label from {labels[node]} to {new_label}")
                labels[node] = new_label
                changes += 1

        print(f"Iteration {iteration + 1} completed. {changes} label changes.\n")

        if changes == 0:
            print("Labels have stabilized. Algorithm converged.\n")
            break

    # Group nodes by final label = detected communities
    communities = {}
    for node, label in labels.items():
        communities.setdefault(label, set()).add(node)

    print("Final communities:")
    for idx, community in enumerate(communities.values()):
        print(f"  Community {idx + 1}: {sorted(community)}")

    return communities

# Run the algorithm
final_communities = manual_label_propagation(G)

# ==============================================================================
# STEP 3: Visualize Final Communities
# ==============================================================================

def draw_communities(G, communities, title):
    """
    Visualizes the network with nodes colored by their final community labels.
    """
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 7))

    # Assign a community color index to each node
    node_color_map = {}
    for idx, community in enumerate(communities.values()):
        for node in community:
            node_color_map[node] = idx

    colors = [node_color_map[node] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.viridis, node_size=500)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos)

    plt.title(title)
    plt.axis('off')
    plt.show()

# Draw the result
draw_communities(G, final_communities, "Final Communities After Label Propagation")
