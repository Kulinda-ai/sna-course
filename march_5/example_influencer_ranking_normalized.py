# FILE: example_influencer_ranking_normalized.py

import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# ==============================================================================
# STEP 1: Load Graph from JSON Files and Build a Weighted Graph
# ==============================================================================

def create_graph_from_json(nodes_file_path, edges_file_path):
    """
    Load nodes and edges from JSON files into a weighted undirected NetworkX graph.
    If edges do not include weights, a default of 1 is used.
    """
    with open(nodes_file_path, 'r') as file:
        nodes_data = json.load(file)

    with open(edges_file_path, 'r') as file:
        edges_data = json.load(file)

    G = nx.Graph()

    for node in nodes_data:
        G.add_node(node['id'], label=node.get('label', ''))

    for edge in edges_data:
        weight = edge.get('weight', 1)
        G.add_edge(edge['source'], edge['target'], weight=weight)

    return G

# Load the graph
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'
G = create_graph_from_json(nodes_file_path, edges_file_path)

# ==============================================================================
# STEP 2: Calculate Centrality Metrics
# ==============================================================================

# These metrics measure different types of importance in a network:
# - Degree: number of connections
# - Eigenvector: how connected your neighbors are
# - Betweenness: control over information flow
# - Closeness: how fast you can reach others

degree_centrality = nx.degree_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)

# ==============================================================================
# STEP 3: Normalize Metrics (Optional for Some, Essential for Others)
# ==============================================================================

def normalize_centrality(centrality_dict):
    """
    Normalize centrality values to range 0-1 using min-max normalization.
    This ensures fair contribution to combined influence score.
    """
    min_val = min(centrality_dict.values())
    max_val = max(centrality_dict.values())
    if max_val - min_val == 0:
        return {node: 0 for node in centrality_dict}
    return {node: (value - min_val) / (max_val - min_val) for node, value in centrality_dict.items()}

# Normalize metrics that vary wildly in scale
betweenness_centrality = normalize_centrality(betweenness_centrality)
closeness_centrality = normalize_centrality(closeness_centrality)

# Degree and eigenvector centrality already tend to be 0–1, but this keeps things consistent

# ==============================================================================
# STEP 4: Optional — Calculate Average Edge Weight per Node
# ==============================================================================

def average_weight(G, node):
    """
    Calculate the average weight of all edges connected to a node.
    Not used in influence score here, but useful for further filtering or features.
    """
    total_weight = sum(weight for _, _, weight in G.edges(node, data='weight'))
    return total_weight / G.degree(node) if G.degree(node) > 0 else 0

# ==============================================================================
# STEP 5: Define and Calculate Influence Scores
# ==============================================================================

# Weights control the contribution of each centrality measure
weights = {
    'degree': 0.4,
    'eigenvector': 0.3,
    'betweenness': 0.2,
    'closeness': 0.1,
}

# Ensure they sum to 1 (can auto-adjust if needed)
total_weight = sum(weights.values())
weights = {k: v / total_weight for k, v in weights.items()}

def influence_score(node):
    """
    Compute a node's influence score using a weighted sum of centralities.
    """
    return (degree_centrality[node] * weights['degree'] +
            eigenvector_centrality[node] * weights['eigenvector'] +
            betweenness_centrality[node] * weights['betweenness'] +
            closeness_centrality[node] * weights['closeness'])

# Calculate influence scores for all nodes
influence_scores = {node: influence_score(node) for node in G.nodes()}

# ==============================================================================
# STEP 6: Rank Nodes by Influence Score
# ==============================================================================

ranked_nodes = sorted(influence_scores, key=influence_scores.get, reverse=True)

print("Nodes ranked from most to least influential:")
for rank, node in enumerate(ranked_nodes, start=1):
    print(f"{rank}. {node} (Score: {influence_scores[node]:.4f})")

# ==============================================================================
# STEP 7: Visualize Network with Node Size Based on Influence
# ==============================================================================

plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, seed=42)

# Scale node sizes based on influence score
node_sizes = [influence_scores[node] * 1000 for node in G.nodes()]

# Draw the network
nx.draw(
    G, pos,
    with_labels=True,
    node_size=node_sizes,
    node_color='skyblue',
    edge_color='gray',
    font_size=10
)

plt.title("Graph with Node Influence Scaling (Size ∝ Influence Score)")
plt.tight_layout()
plt.show()
