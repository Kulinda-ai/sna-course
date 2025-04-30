# FILE: initial_analysis.py

import json
import networkx as nx

# === Step 1: Load data ===
# Replace this with your actual file or data source
with open('networkx.json', 'r') as f:
    graph_data = json.load(f)

# === Step 2: Create a directed graph ===
G = nx.DiGraph()

# Add nodes with attributes
for node in graph_data['nodes']:
    data = node['data']
    G.add_node(data['id'], **data)

# Add edges with attributes
for edge in graph_data['edges']:
    data = edge['data']
    G.add_edge(data['source'], data['target'], **data)

print(f"Initial graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# === Step 3: Filter out 'dust' transactions ===
# We'll define 'dust' as either explicitly marked or below a small threshold
dust_threshold = 0.00001  # You can adjust this as needed

edges_to_remove = [
    (u, v) for u, v, d in G.edges(data=True)
    if d.get('dust') is True or d.get('value', 0) <= dust_threshold
]

G.remove_edges_from(edges_to_remove)
print(f"After dust filtering: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# === Step 4: Calculate Social Network Metrics ===

# Degree (total number of connections)
degree_dict = dict(G.degree())
nx.set_node_attributes(G, degree_dict, 'degree')

# In-Degree (received transactions)
in_degree_dict = dict(G.in_degree())
nx.set_node_attributes(G, in_degree_dict, 'in_degree')

# Out-Degree (sent transactions)
out_degree_dict = dict(G.out_degree())
nx.set_node_attributes(G, out_degree_dict, 'out_degree')

# Degree Centrality (normalized influence based on degree)
degree_centrality_dict = nx.degree_centrality(G)
nx.set_node_attributes(G, degree_centrality_dict, 'degree_centrality')

# Betweenness Centrality (nodes that act as bridges)
betweenness_dict = nx.betweenness_centrality(G)
nx.set_node_attributes(G, betweenness_dict, 'betweenness')

# PageRank (influence based on connected structure)
pagerank_dict = nx.pagerank(G, alpha=0.85)
nx.set_node_attributes(G, pagerank_dict, 'pagerank')

# Closeness Centrality (how close a node is to all others)
closeness_dict = nx.closeness_centrality(G)
nx.set_node_attributes(G, closeness_dict, 'closeness')

# === Step 5: Output a sample of enriched node data ===
print("\nSample node with calculated metrics:")
for node, data in list(G.nodes(data=True))[:1]:
    for k, v in data.items():
        print(f"{k}: {v}")

# Export to JSON (like original format, but now enriched with metrics)
enriched_nodes = [{"data": dict(G.nodes[n])} for n in G.nodes]
enriched_edges = [{"data": d} for _, _, d in G.edges(data=True)]

output = {
    "nodes": enriched_nodes,
    "edges": enriched_edges
}

with open('initial_analysis.json', 'w') as f:
    json.dump(output, f, indent=2)