# FILE: example_at_risk_identification.py

import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# ==============================================================================
# STEP 1: Load nodes and edges, and build a weighted undirected graph
# ==============================================================================

def create_graph_from_json(nodes_file_path, edges_file_path):
    with open(nodes_file_path, 'r') as file:
        nodes_data = json.load(file)
    with open(edges_file_path, 'r') as file:
        edges_data = json.load(file)

    G = nx.Graph()

    # Add nodes with optional labels
    for node in nodes_data:
        G.add_node(node['id'], label=node.get('label', ''))

    # Add edges with optional weights (default = 1)
    for edge in edges_data:
        weight = edge.get('weight', 1)
        G.add_edge(edge['source'], edge['target'], weight=weight)

    return G

# Load graph from JSON files
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'
G = create_graph_from_json(nodes_file_path, edges_file_path)

# ==============================================================================
# STEP 2: Compute core centrality metrics for influence estimation
# ==============================================================================

degree_centrality = nx.degree_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)

# ==============================================================================
# STEP 3: Normalize betweenness and closeness so they are on the same scale
# ==============================================================================

def normalize_centrality(centrality_dict):
    min_val = min(centrality_dict.values())
    max_val = max(centrality_dict.values())
    if max_val - min_val == 0:
        return {node: 0 for node in centrality_dict}
    return {node: (value - min_val) / (max_val - min_val) for node, value in centrality_dict.items()}

betweenness_centrality = normalize_centrality(betweenness_centrality)
closeness_centrality = normalize_centrality(closeness_centrality)

# ==============================================================================
# STEP 4: Compute an "influence score" for each node using weighted centralities
# ==============================================================================

# These weights reflect how much each centrality contributes to "influence"
weights = {'degree': 0.4, 'eigenvector': 0.3, 'betweenness': 0.2, 'closeness': 0.1}
# Normalize the weights (in case someone edits them and they don't sum to 1)
total_weight = sum(weights.values())
weights = {k: v / total_weight for k, v in weights.items()}

def influence_score(node):
    return (degree_centrality[node] * weights['degree'] +
            eigenvector_centrality[node] * weights['eigenvector'] +
            betweenness_centrality[node] * weights['betweenness'] +
            closeness_centrality[node] * weights['closeness'])

influence_scores = {node: influence_score(node) for node in G.nodes()}

# ==============================================================================
# STEP 5: Identify "influencers" (top 10% most influential nodes)
# ==============================================================================

num_influencers = max(1, int(len(G.nodes()) * 0.1))  # Ensure at least one
sorted_nodes = sorted(influence_scores, key=influence_scores.get, reverse=True)
influencers = set(sorted_nodes[:num_influencers])

print(f"Top {num_influencers} Influencers Identified:")
for rank, node in enumerate(sorted_nodes[:num_influencers], start=1):
    print(f"{rank}. {node} (Score: {influence_scores[node]:.4f})")

# ==============================================================================
# STEP 6: Identify At-Risk Nodes
# ==============================================================================

mildly_at_risk = []   # Connected to exactly 1 influencer
highly_at_risk = []   # Connected to 2+ influencers

for node in G.nodes():
    if node in influencers:
        continue  # Skip influencer nodes
    
    neighbors = set(G.neighbors(node))
    influencer_neighbors = neighbors.intersection(influencers)

    if len(influencer_neighbors) == 1:
        mildly_at_risk.append(node)
    elif len(influencer_neighbors) > 1:
        highly_at_risk.append(node)

# ==============================================================================
# STEP 7: Print At-Risk Results
# ==============================================================================

print("\nMildly At-Risk Accounts (Connected to 1 Influencer):")
for node in mildly_at_risk:
    print(f"Node {node}: Connected to 1 Influencer")

print("\nHighly At-Risk Accounts (Connected to Multiple Influencers):")
for node in highly_at_risk:
    print(f"Node {node}: Connected to {len(set(G.neighbors(node)).intersection(influencers))} Influencers")

# ==============================================================================
# STEP 8: Visualize the Graph with Risk/Influence Categories
# ==============================================================================

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.3, iterations=50)  # Force-directed layout

# Assign colors by role
node_colors = []
for node in G.nodes():
    if node in influencers:
        node_colors.append('red')       # Influencers
    elif node in mildly_at_risk:
        node_colors.append('orange')    # Mildly at-risk
    elif node in highly_at_risk:
        node_colors.append('purple')    # Highly at-risk
    else:
        node_colors.append('skyblue')   # Others

# Scale node size by influence
node_sizes = [influence_scores[node] * 1000 for node in G.nodes()]

# Draw the graph
nx.draw(G, pos, with_labels=True, node_size=node_sizes,
        node_color=node_colors, edge_color='gray', font_size=8)

plt.title("Graph Visualization — Red = Influencers, Orange = Mildly At-Risk, Purple = Highly At-Risk")
plt.show()
