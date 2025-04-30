# FILE: centrality_comparison.py

import json
import networkx as nx
import csv

# === Load raw data ===
with open('networkx.json', 'r') as f:
    raw_data = json.load(f)

# Dust filter threshold
dust_threshold = 0.00001

# === Helper: Build graph from raw JSON ===
def build_graph(use_weights=False):
    G = nx.DiGraph()

    # Add all nodes
    for node in raw_data['nodes']:
        G.add_node(node['data']['id'], **node['data'])

    # Add edges
    for edge in raw_data['edges']:
        edge_data = edge['data']
        source = edge_data['source']
        target = edge_data['target']
        value = edge_data.get('value', 0)

        if edge_data.get('dust') is True or value <= dust_threshold:
            continue  # Skip dust

        attrs = {'value': value}
        if use_weights:
            attrs['weight'] = value

        G.add_edge(source, target, **attrs)

    # Remove nodes with no edges
    nodes_with_edges = set()
    for u, v in G.edges():
        nodes_with_edges.add(u)
        nodes_with_edges.add(v)

    for node in list(G.nodes):
        if node not in nodes_with_edges:
            G.remove_node(node)

    return G

# === Build both graphs ===
G_unweighted = build_graph(use_weights=False)
G_weighted = build_graph(use_weights=True)

print(f"Unweighted: {G_unweighted.number_of_nodes()} nodes, {G_unweighted.number_of_edges()} edges")
print(f"Weighted:   {G_weighted.number_of_nodes()} nodes, {G_weighted.number_of_edges()} edges")

# === Compute centralities ===

# Unweighted
pagerank_un = nx.pagerank(G_unweighted)
betweenness_un = nx.betweenness_centrality(G_unweighted)

# Weighted
pagerank_wt = nx.pagerank(G_weighted, weight='weight')
betweenness_wt = nx.betweenness_centrality(G_weighted, weight='weight')

# === Join & Export ===

all_ids = set(G_unweighted.nodes()).union(set(G_weighted.nodes()))
rows = []

for node_id in all_ids:
    row = {
        'account_id': node_id,
        'wallet_type': G_unweighted.nodes[node_id].get('wallet_type', G_weighted.nodes[node_id].get('wallet_type')),
        'pagerank_unweighted': pagerank_un.get(node_id, 0),
        'pagerank_weighted': pagerank_wt.get(node_id, 0),
        'betweenness_unweighted': betweenness_un.get(node_id, 0),
        'betweenness_weighted': betweenness_wt.get(node_id, 0),
    }
    rows.append(row)

# Write to CSV
with open('centrality_comparison.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'account_id',
        'wallet_type',
        'pagerank_unweighted', 'pagerank_weighted',
        'betweenness_unweighted', 'betweenness_weighted'
    ])
    writer.writeheader()
    writer.writerows(rows)

print("✅ Exported centrality_comparison.csv with both weighted and unweighted metrics.")
