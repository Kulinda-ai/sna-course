# FILE: example_network_metrics.py

import json
import networkx as nx
import pandas as pd
import numpy as np

# ==============================================================================
# STEP 1: Utility — Normalize centrality scores between 0 and 1
# ==============================================================================

def normalize_centrality(centrality_dict):
    """
    Normalizes a centrality dictionary using min-max scaling.
    This makes metrics comparable when combined into influence scores.
    """
    min_val = min(centrality_dict.values())
    max_val = max(centrality_dict.values())
    if max_val - min_val == 0:
        return {node: 0 for node in centrality_dict}
    return {node: (value - min_val) / (max_val - min_val) for node, value in centrality_dict.items()}

# ==============================================================================
# STEP 2: Analyze Full Network — Summary Stats + Node Scores
# ==============================================================================

def analyze_network(G):
    # -------------------------
    # A. Graph-level statistics
    # -------------------------

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)

    # Compute average shortest path length
    avg_shortest_path_length = (
        nx.average_shortest_path_length(G, weight='weight') if nx.is_connected(G)
        else float('nan')
    )

    avg_clustering = nx.average_clustering(G)
    global_betweenness = np.mean(list(nx.betweenness_centrality(G).values()))
    global_closeness = np.mean(list(nx.closeness_centrality(G).values()))

    network_info = pd.DataFrame({
        'Metric': [
            'Number of Nodes', 'Number of Edges', 'Density',
            'Average Shortest Path Length', 'Average Clustering',
            'Global Betweenness Centrality', 'Global Closeness Centrality'
        ],
        'Value': [
            num_nodes, num_edges, density, avg_shortest_path_length,
            avg_clustering, global_betweenness, global_closeness
        ]
    })

    network_info_json = {
        'Number of Nodes': num_nodes,
        'Number of Edges': num_edges,
        'Density': density,
        'Average Shortest Path Length': avg_shortest_path_length,
        'Average Clustering': avg_clustering,
        'Global Betweenness Centrality': global_betweenness,
        'Global Closeness Centrality': global_closeness,
    }

    # -------------------------
    # B. Node-level centrality metrics
    # -------------------------

    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    degree_centrality = nx.degree_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6)
    degrees = dict(G.degree())

    # Normalize selected metrics for comparability
    betweenness = normalize_centrality(betweenness)
    closeness = normalize_centrality(closeness)

    # -------------------------
    # C. Eccentricity: Distance to farthest node
    # -------------------------

    if nx.is_connected(G):
        eccentricity = nx.eccentricity(G)
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
        eccentricity = nx.eccentricity(subgraph)
        for node in set(G.nodes()) - set(largest_cc):
            eccentricity[node] = None

    # -------------------------
    # D. Combine metrics into an influence score
    # -------------------------

    weights = {
        'degree': 0.4,
        'eigenvector': 0.3,
        'betweenness': 0.2,
        'closeness': 0.1
    }

    # Ensure weights sum to 1
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    influencer_scores = {
        node: (
            degree_centrality[node] * weights['degree'] +
            eigenvector_centrality[node] * weights['eigenvector'] +
            betweenness[node] * weights['betweenness'] +
            closeness[node] * weights['closeness']
        )
        for node in G.nodes()
    }

    # -------------------------
    # E. Assemble node-level summary table
    # -------------------------

    nodes_info = pd.DataFrame({
        'Node': list(G.nodes()),
        'Betweenness': [betweenness[node] for node in G.nodes()],
        'Closeness': [closeness[node] for node in G.nodes()],
        'Eccentricity': [eccentricity[node] for node in G.nodes()],
        'Degree Centrality': [degree_centrality[node] for node in G.nodes()],
        'Eigenvector Centrality': [eigenvector_centrality[node] for node in G.nodes()],
        'Degrees': [degrees[node] for node in G.nodes()],
        'Influencer Score': [influencer_scores[node] for node in G.nodes()]
    })

    # Rank nodes by influence
    nodes_info['Rank'] = nodes_info['Influencer Score'].rank(ascending=False)

    # Sort and convert to JSON
    nodes_info_sorted = nodes_info.sort_values(by='Influencer Score', ascending=False).reset_index(drop=True)
    nodes_info_json = nodes_info_sorted.to_json(orient='records')

    # Save both graph-level and node-level summaries
    with open('network_info.json', 'w') as f:
        json.dump(network_info_json, f, indent=4)

    with open('nodes_info.json', 'w') as f:
        json.dump(json.loads(nodes_info_json), f, indent=4)

    return network_info, nodes_info_sorted, network_info_json, nodes_info_json

# ==============================================================================
# STEP 3: Load the Graph and Run Analysis
# ==============================================================================

def create_graph_from_json(nodes_file_path, edges_file_path):
    """
    Load graph structure from JSON files.
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

# Set file paths
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Run analysis
G = create_graph_from_json(nodes_file_path, edges_file_path)
network_info, nodes_info_sorted, network_info_json, nodes_info_json = analyze_network(G)

# ==============================================================================
# STEP 4: Output Preview
# ==============================================================================

# Print overall network statistics
print("Network Information:")
print(network_info)

# Print sorted node influence rankings
print("\nRanked Nodes Information:")
print(nodes_info_sorted)

# Show first part of JSON strings
print("\nNetwork Information (JSON):")
print(json.dumps(network_info_json, indent=4)[:100] + '...')

print("\nNodes Information (JSON):")
print(nodes_info_json[:100] + '...')
