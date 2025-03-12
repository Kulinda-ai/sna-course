import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Create a weighted graph
def create_graph_from_json(nodes_file_path, edges_file_path):
    # Load nodes
    with open(nodes_file_path, 'r') as file:
        nodes_data = json.load(file)
    
    # Load edges
    with open(edges_file_path, 'r') as file:
        edges_data = json.load(file)
    
    # Create a new graph
    G = nx.Graph()
    
    # Add nodes to the graph
    for node in nodes_data:
        G.add_node(node['id'], label=node.get('label', ''))
    
    # Add edges with weights if available
    for edge in edges_data:
        weight = edge.get('weight', 1)  # Default weight to 1 if not specified
        G.add_edge(edge['source'], edge['target'], weight=weight)
    
    return G

# Replace 'networkx_nodes.json' and 'networkx_edges.json' with the actual paths
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Create the graph
G = create_graph_from_json(nodes_file_path, edges_file_path)

# Step 2: Calculate centrality measures
degree_centrality = nx.degree_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)

# Step 3: Normalize centrality measures that need it
def normalize_centrality(centrality_dict):
    """ Min-max normalization of centrality values """
    min_val = min(centrality_dict.values())
    max_val = max(centrality_dict.values())
    if max_val - min_val == 0:  # Avoid division by zero if all values are the same
        return {node: 0 for node in centrality_dict}
    return {node: (value - min_val) / (max_val - min_val) for node, value in centrality_dict.items()}

# Normalize betweenness and closeness centrality
betweenness_centrality = normalize_centrality(betweenness_centrality)
closeness_centrality = normalize_centrality(closeness_centrality)

# Step 4: Analyze connection quality (average weight)
def average_weight(G, node):
    """ Calculate the average weight of edges connected to a node """
    total_weight = sum(weight for _, _, weight in G.edges(node, data='weight'))
    return total_weight / G.degree(node) if G.degree(node) > 0 else 0

# Step 5: Influence Score Calculation
# Customizable weights for each centrality metric (must sum to 1)
weights = {
    'degree': 0.4,
    'eigenvector': 0.3,
    'betweenness': 0.2,
    'closeness': 0.1,
}

# Auto-normalize weights if necessary
total_weight = sum(weights.values())
weights = {k: v / total_weight for k, v in weights.items()}  # Ensure sum = 1

def influence_score(node):
    """ Compute influence score as weighted sum of normalized centrality measures """
    return (degree_centrality[node] * weights['degree'] +
            eigenvector_centrality[node] * weights['eigenvector'] +
            betweenness_centrality[node] * weights['betweenness'] +
            closeness_centrality[node] * weights['closeness'])

# Compute influence scores for all nodes
influence_scores = {node: influence_score(node) for node in G.nodes()}

# Step 6: Rank nodes from most to least influential
ranked_nodes = sorted(influence_scores, key=influence_scores.get, reverse=True)

# Print ranked list of nodes
print("Nodes ranked from most to least influential:")
for rank, node in enumerate(ranked_nodes, start=1):
    print(f"{rank}. {node} (Score: {influence_scores[node]:.4f})")

# Step 7: Visualization (Optional)
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G)  # Layout for visualization
node_sizes = [influence_scores[node] * 1000 for node in G.nodes()]  # Scale sizes

nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color='skyblue', edge_color='gray', font_size=10)
plt.title("Graph with Node Influence Scaling")
plt.show()
