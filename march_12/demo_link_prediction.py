# FILE: demo_link_prediction.py

import json
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.link_prediction import (
    jaccard_coefficient,
    adamic_adar_index,
    preferential_attachment,
    resource_allocation_index
)

# ==============================================================================
# STEP 1: Load Graph from JSON
# ==============================================================================

def create_graph_from_json(nodes_file_path, edges_file_path):
    """
    Loads nodes and edges from JSON and returns an undirected NetworkX graph.
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
# STEP 2: Visualize Graph with Predicted Edges
# ==============================================================================

def draw_predicted_links(G, predicted_links, title):
    """
    Draws the original graph and overlays predicted edges in red (dashed).
    """
    pos = nx.spring_layout(G, seed=42, k=0.8, iterations=100)

    plt.figure(figsize=(14, 10))
    nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color='gray')

    # Draw predicted edges (e.g., future links)
    predicted_edges = [(u, v) for u, v, _ in predicted_links]
    predicted_nodes = {n for edge in predicted_edges for n in edge}

    nx.draw_networkx_edges(
        G, pos,
        edgelist=predicted_edges,
        edge_color='red',
        style='dashed',
        width=4
    )

    node_sizes = [1000 if node in predicted_nodes else 500 for node in G.nodes()]
    node_colors = ['orange' if node in predicted_nodes else 'lightblue' for node in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.show()

# ==============================================================================
# STEP 3: Prediction Utilities
# ==============================================================================

def get_top_predictions(predictions, top_n=10):
    return sorted(predictions, key=lambda x: x[2], reverse=True)[:top_n]

def common_neighbors_prediction(G, top_n=10):
    """
    Predict edges using the number of common neighbors (simplest approach).
    """
    preds = []
    nodes = list(G.nodes())

    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            if not G.has_edge(u, v):
                cn = len(list(nx.common_neighbors(G, u, v)))
                if cn > 0:
                    preds.append((u, v, cn))

    return sorted(preds, key=lambda x: x[2], reverse=True)[:top_n]

# ==============================================================================
# STEP 4: Print Predictions for Each Algorithm
# ==============================================================================

def print_common_neighbors_predictions(G, predictions):
    print("\n=== Common Neighbors Predictions ===\n")
    for idx, (u, v, score) in enumerate(predictions, 1):
        common = sorted(nx.common_neighbors(G, u, v))
        print(f"Prediction #{idx}:")
        print(f"Node {u} <--> Node {v}")
        print(f"Reason: {len(common)} shared neighbors → {common}\n")

def print_jaccard_predictions(G, predictions):
    print("\n=== Jaccard Coefficient Predictions ===\n")
    for idx, (u, v, score) in enumerate(predictions, 1):
        u_neighbors = set(G.neighbors(u))
        v_neighbors = set(G.neighbors(v))
        union_size = len(u_neighbors.union(v_neighbors))
        intersection_size = len(u_neighbors.intersection(v_neighbors))

        print(f"Prediction #{idx}:")
        print(f"Node {u} <--> Node {v}")
        print(f"Reason: {intersection_size}/{union_size} overlap → Jaccard Score: {score:.4f}\n")

def print_adamic_adar_predictions(G, predictions):
    print("\n=== Adamic-Adar Index Predictions ===\n")
    for idx, (u, v, score) in enumerate(predictions, 1):
        common = sorted(nx.common_neighbors(G, u, v))
        print(f"Prediction #{idx}:")
        print(f"Node {u} <--> Node {v}")
        print(f"Reason: Weighted by inverse log degree of common neighbors → Score: {score:.4f}\n")

def print_preferential_attachment_predictions(G, predictions):
    print("\n=== Preferential Attachment Predictions ===\n")
    for idx, (u, v, score) in enumerate(predictions, 1):
        print(f"Prediction #{idx}:")
        print(f"Node {u} (deg={G.degree(u)}) <--> Node {v} (deg={G.degree(v)})")
        print(f"Reason: Score = {score} = deg(u) × deg(v)\n")

def print_resource_allocation_predictions(G, predictions):
    print("\n=== Resource Allocation Index Predictions ===\n")
    for idx, (u, v, score) in enumerate(predictions, 1):
        common = sorted(nx.common_neighbors(G, u, v))
        print(f"Prediction #{idx}:")
        print(f"Node {u} <--> Node {v}")
        print(f"Reason: Score based on common neighbors with low degree → Score: {score:.4f}\n")

# ==============================================================================
# STEP 5: Run All Prediction Methods and Print Results
# ==============================================================================

nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'
G = create_graph_from_json(nodes_file_path, edges_file_path)

# 1. Common Neighbors
top_cn = common_neighbors_prediction(G, top_n=10)
print_common_neighbors_predictions(G, top_cn)
draw_predicted_links(G, top_cn, "Common Neighbors Link Prediction")

# 2. Jaccard Coefficient
jc_predictions = list(jaccard_coefficient(G))
top_jc = get_top_predictions(jc_predictions, top_n=10)
print_jaccard_predictions(G, top_jc)
draw_predicted_links(G, top_jc, "Jaccard Coefficient Link Prediction")

# 3. Adamic-Adar Index
aa_predictions = list(adamic_adar_index(G))
top_aa = get_top_predictions(aa_predictions, top_n=10)
print_adamic_adar_predictions(G, top_aa)
draw_predicted_links(G, top_aa, "Adamic-Adar Index Link Prediction")

# 4. Preferential Attachment
pa_predictions = list(preferential_attachment(G))
top_pa = get_top_predictions(pa_predictions, top_n=10)
print_preferential_attachment_predictions(G, top_pa)
draw_predicted_links(G, top_pa, "Preferential Attachment Link Prediction")

# 5. Resource Allocation Index
ra_predictions = list(resource_allocation_index(G))
top_ra = get_top_predictions(ra_predictions, top_n=10)
print_resource_allocation_predictions(G, top_ra)
draw_predicted_links(G, top_ra, "Resource Allocation Link Prediction")

# ==============================================================================
# STEP 6: Summary Table of Top Predictions
# ==============================================================================

def summary_table(predictions_dict):
    print("\n=== SUMMARY OF PREDICTED LINKS ===\n")
    print(f"{'Method':<30} | Predicted Links")
    print("-" * 70)
    for method, predictions in predictions_dict.items():
        pairs = [f"({u}, {v})" for u, v, _ in predictions]
        print(f"{method:<30} | {', '.join(pairs)}")

predictions_summary = {
    "Common Neighbors": top_cn,
    "Jaccard Coefficient": top_jc,
    "Adamic-Adar Index": top_aa,
    "Preferential Attachment": top_pa,
    "Resource Allocation Index": top_ra
}

summary_table(predictions_summary)
