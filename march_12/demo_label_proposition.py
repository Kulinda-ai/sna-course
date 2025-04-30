# FILE: demo_label_proposition.py

import json
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import random

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

# Paths to your JSON files
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Create the graph
G = create_graph_from_json(nodes_file_path, edges_file_path)

# Manual label propagation algorithm with step-by-step label updates
def manual_label_propagation(G, max_iterations=10):
    # Initialize each node's label to its own node ID
    labels = {node: node for node in G.nodes()}

    print("Initial labels:")
    for node, label in labels.items():
        print(f"  Node {node}: Label {label}")
    print("\n")

    # Perform label propagation
    for iteration in range(max_iterations):
        print(f"=== Iteration {iteration + 1} ===")
        nodes = list(G.nodes())
        random.shuffle(nodes)  # Randomize node order for each iteration
        
        changes = 0  # Track label changes in this iteration
        
        for node in nodes:
            neighbor_labels = [labels[neighbor] for neighbor in G.neighbors(node)]
            
            if not neighbor_labels:
                continue  # No neighbors, skip
            
            # Find the most common label(s)
            label_count = Counter(neighbor_labels)
            max_count = max(label_count.values())
            most_common_labels = [label for label, count in label_count.items() if count == max_count]
            
            # Randomly choose among equally frequent labels
            new_label = random.choice(most_common_labels)
            
            # If the label has changed, update and print
            if labels[node] != new_label:
                print(f"  Node {node} changed label from {labels[node]} to {new_label}")
                labels[node] = new_label
                changes += 1
        
        print(f"Iteration {iteration + 1} completed. {changes} label changes.\n")
        
        # If no label changed, we have converged
        if changes == 0:
            print("Labels have stabilized. Algorithm converged.\n")
            break

    # Group nodes by labels (communities)
    communities = {}
    for node, label in labels.items():
        communities.setdefault(label, set()).add(node)

    print("Final communities:")
    for idx, community in enumerate(communities.values()):
        print(f"  Community {idx + 1}: {sorted(community)}")
    
    return communities

# Run the manual label propagation
final_communities = manual_label_propagation(G)

# Visualize the final communities
def draw_communities(G, communities, title):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 7))

    # Flatten communities into node -> community index
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

draw_communities(G, final_communities, "Final Communities After Label Propagation")
