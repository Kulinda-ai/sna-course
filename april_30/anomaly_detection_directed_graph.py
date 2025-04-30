# FILE: anomaly_detection_directed_graph.py

# anomaly_detection_directed_graph.py

import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np

# === Load Nodes and Edges ===
with open('networkx_nodes.json') as f:
    node_data = json.load(f)

with open('networkx_edges.json') as f:
    edge_data = json.load(f)

# === Fix Labels
for node in node_data:
    node['label'] = f"node_{node['id']}"

# === Create Directed Graph ===
G = nx.DiGraph()
for node in node_data:
    G.add_node(str(node['id']), label=node['label'])

for edge in edge_data:
    G.add_edge(str(edge['source']), str(edge['target']))

# === Compute Node Features ===
print("Computing network features...")
betweenness = nx.betweenness_centrality(G)
pagerank = nx.pagerank(G)
clustering = nx.clustering(G.to_undirected())

features = []
for node in G.nodes():
    features.append({
        'node': node,
        'degree': G.degree(node),
        'in_degree': G.in_degree(node),
        'out_degree': G.out_degree(node),
        'clustering': clustering.get(node, 0),
        'betweenness': betweenness.get(node, 0),
        'pagerank': pagerank.get(node, 0),
    })

df = pd.DataFrame(features)

# === Compute Z-scores for each feature ===
print("Computing z-scores...")
z_scores = (df.drop(columns=['node']) - df.drop(columns=['node']).mean()) / df.drop(columns=['node']).std()
z_scores.columns = [f"{col}_zscore" for col in z_scores.columns]
df = pd.concat([df, z_scores], axis=1)

# === Normalize & Detect Anomalies ===
print("Running Isolation Forest...")
X = df[[col for col in df.columns if not col.startswith('node') and not col.endswith('_zscore')]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(contamination=0.05, random_state=42)
df['anomaly_score'] = model.fit_predict(X_scaled)

# === Split Nodes ===
anomalous_nodes_df = df[df['anomaly_score'] == -1]
normal_nodes_df = df[df['anomaly_score'] == 1]

print(f"\nTotal nodes: {len(df)}")
print(f"Normal nodes: {len(normal_nodes_df)}")
print(f"Anomalous nodes: {len(anomalous_nodes_df)}")

# === Detailed Output for Anomalous Nodes ===
print("\nDetailed Feature and Z-Score Breakdown for Anomalous Nodes:")
cols_to_show = ['node', 'degree', 'in_degree', 'out_degree', 'clustering', 'betweenness', 'pagerank',
                'degree_zscore', 'in_degree_zscore', 'out_degree_zscore',
                'clustering_zscore', 'betweenness_zscore', 'pagerank_zscore']
print(anomalous_nodes_df[cols_to_show].sort_values(by='pagerank', ascending=False).to_string(index=False))

# === Visualize the Graph with Labels ===
print("\nVisualizing anomalies...")
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(14, 10))
nx.draw_networkx_edges(G, pos, alpha=0.2)
nx.draw_networkx_nodes(G, pos, nodelist=normal_nodes_df['node'].tolist(), node_color='lightgray', node_size=40, label='Normal')
nx.draw_networkx_nodes(G, pos, nodelist=anomalous_nodes_df['node'].tolist(), node_color='red', node_size=80, label='Anomalous')

# Add node labels to anomalies
labels = {node: f"n{node}" for node in anomalous_nodes_df['node'].tolist()}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='black')

plt.title("Anomaly Detection in Network (Isolation Forest)", fontsize=14)
plt.legend(scatterpoints=1, loc='best')
plt.axis('off')
plt.tight_layout()
plt.show()
