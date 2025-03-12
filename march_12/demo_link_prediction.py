import json
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.link_prediction import (
    jaccard_coefficient,
    adamic_adar_index,
    preferential_attachment,
    resource_allocation_index
)

# === Load Graph from JSON ===
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

# === Draw Network with Predicted Links ===
def draw_predicted_links(G, predicted_links, title):
    pos = nx.spring_layout(G, seed=42, k=0.8, iterations=100)

    plt.figure(figsize=(14, 10))

    nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color='gray')

    predicted_edges = [(u, v) for u, v, score in predicted_links]
    predicted_nodes = set([node for edge in predicted_edges for node in edge])

    nx.draw_networkx_edges(
        G, pos,
        edgelist=predicted_edges,
        edge_color='red',
        style='dashed',
        width=4
    )

    node_sizes = []
    node_colors = []
    for node in G.nodes():
        if node in predicted_nodes:
            node_sizes.append(1000)
            node_colors.append('orange')
        else:
            node_sizes.append(500)
            node_colors.append('lightblue')

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, edgecolors='black')

    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.show()

# === Get Top N Predictions ===
def get_top_predictions(predictions, top_n=10):
    return sorted(predictions, key=lambda x: x[2], reverse=True)[:top_n]

# === Common Neighbors Prediction + Pretty Print ===
def common_neighbors_prediction(G, top_n=10):
    preds = []
    nodes = list(G.nodes())

    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            if not G.has_edge(u, v):
                cn = len(list(nx.common_neighbors(G, u, v)))
                if cn > 0:
                    preds.append((u, v, cn))

    return sorted(preds, key=lambda x: x[2], reverse=True)[:top_n]

def print_common_neighbors_predictions(G, predictions):
    print("\n=== Common Neighbors Predictions ===\n")
    for idx, (u, v, score) in enumerate(predictions, 1):
        u_neighbors = sorted(list(G.neighbors(u)))
        v_neighbors = sorted(list(G.neighbors(v)))
        common = sorted(list(nx.common_neighbors(G, u, v)))

        print(f"Prediction #{idx}:")
        print(f"Node {u} (neighbors: {u_neighbors}) <--> Node {v} (neighbors: {v_neighbors})")
        print(f"Reason: They share {score} common neighbors -> {common}\n")

def print_jaccard_predictions(G, predictions):
    print("\n=== Jaccard Coefficient Predictions ===\n")
    for idx, (u, v, score) in enumerate(predictions, 1):
        u_neighbors = set(G.neighbors(u))
        v_neighbors = set(G.neighbors(v))
        union_size = len(u_neighbors.union(v_neighbors))
        intersection_size = len(u_neighbors.intersection(v_neighbors))

        print(f"Prediction #{idx}:")
        print(f"Node {u} (neighbors: {sorted(u_neighbors)}) <--> Node {v} (neighbors: {sorted(v_neighbors)})")
        print(f"Reason: {intersection_size} shared neighbors out of {union_size} total neighbors (Jaccard Score: {score:.4f})\n")

def print_adamic_adar_predictions(G, predictions):
    print("\n=== Adamic-Adar Index Predictions ===\n")
    for idx, (u, v, score) in enumerate(predictions, 1):
        common = sorted(list(nx.common_neighbors(G, u, v)))

        print(f"Prediction #{idx}:")
        print(f"Node {u} <--> Node {v}")
        print(f"Reason: They share {len(common)} common neighbors {common}, weighted by inverse log degree (Score: {score:.4f})\n")

def print_preferential_attachment_predictions(G, predictions):
    print("\n=== Preferential Attachment Predictions ===\n")
    for idx, (u, v, score) in enumerate(predictions, 1):
        deg_u = G.degree(u)
        deg_v = G.degree(v)

        print(f"Prediction #{idx}:")
        print(f"Node {u} (degree {deg_u}) <--> Node {v} (degree {deg_v})")
        print(f"Reason: Preferential Attachment score = {score} (degree(u) * degree(v))\n")

def print_resource_allocation_predictions(G, predictions):
    print("\n=== Resource Allocation Index Predictions ===\n")
    for idx, (u, v, score) in enumerate(predictions, 1):
        common = sorted(list(nx.common_neighbors(G, u, v)))

        print(f"Prediction #{idx}:")
        print(f"Node {u} <--> Node {v}")
        print(f"Reason: Shared neighbors {common}, with resources allocated inversely by their degree (Score: {score:.4f})\n")

# === MAIN RUN ===

nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

G = create_graph_from_json(nodes_file_path, edges_file_path)

# --- COMMON NEIGHBORS ---
top_cn = common_neighbors_prediction(G, top_n=10)
print_common_neighbors_predictions(G, top_cn)
draw_predicted_links(G, top_cn, "Common Neighbors Link Prediction")

# --- JACCARD ---
jc_predictions = list(jaccard_coefficient(G))
top_jc = get_top_predictions(jc_predictions, top_n=10)
print_jaccard_predictions(G, top_jc)
draw_predicted_links(G, top_jc, "Jaccard Coefficient Link Prediction")

# --- ADAMIC-ADAR ---
aa_predictions = list(adamic_adar_index(G))
top_aa = get_top_predictions(aa_predictions, top_n=10)
print_adamic_adar_predictions(G, top_aa)
draw_predicted_links(G, top_aa, "Adamic-Adar Index Link Prediction")

# --- PREFERENTIAL ATTACHMENT ---
pa_predictions = list(preferential_attachment(G))
top_pa = get_top_predictions(pa_predictions, top_n=10)
print_preferential_attachment_predictions(G, top_pa)
draw_predicted_links(G, top_pa, "Preferential Attachment Link Prediction")

# --- RESOURCE ALLOCATION INDEX ---
ra_predictions = list(resource_allocation_index(G))
top_ra = get_top_predictions(ra_predictions, top_n=10)
print_resource_allocation_predictions(G, top_ra)
draw_predicted_links(G, top_ra, "Resource Allocation Link Prediction")

# === SUMMARY TABLE ===
def summary_table(predictions_dict):
    print("\n=== SUMMARY OF PREDICTED LINKS ===\n")
    print(f"{'Method':<30} | Predicted Links")
    print("-" * 70)
    for method, predictions in predictions_dict.items():
        pairs = [f"({u}, {v})" for u, v, _ in predictions]
        print(f"{method:<30} | {', '.join(pairs)}")

# Create summary dictionary
predictions_summary = {
    "Common Neighbors": top_cn,
    "Jaccard Coefficient": top_jc,
    "Adamic-Adar Index": top_aa,
    "Preferential Attachment": top_pa,
    "Resource Allocation Index": top_ra
}

# Print the summary table
summary_table(predictions_summary)
