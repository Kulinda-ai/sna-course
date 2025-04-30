# FILE: anomaly_detection_undirected_graph.py

import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np

# ==============================================================================
# STEP 1: Load Graph Data from JSON
# ==============================================================================

with open('networkx_nodes.json') as f:
    node_data = json.load(f)

with open('networkx_edges.json') as f:
    edge_data = json.load(f)

# Optional cleanup: ensure all nodes have consistent labels
for node in node_data:
    node['label'] = f"node_{node['id']}"

# ==============================================================================
# STEP 2: Create the Undirected Graph
# ==============================================================================

G = nx.Graph()  # Undirected graph
for node in node_data:
    G.add_node(str(node['id']), label=node['label'])

for edge in edge_data:
    G.add_edge(str(edge['source']), str(edge['target']))

# ==============================================================================
# STEP 3: Compute Centrality and Structural Features
# ==============================================================================

print("Computing network features...")

# NetworkX works natively on undirected graphs for most metrics
betweenness = nx.betweenness_centrality(G)
pagerank = nx.pagerank(G)
clustering = nx.clustering(G)

# Collect structural features for each node
features = []
for node in G.nodes():
    features.append({
        'node': node,
        'degree': G.degree(node),
        'clustering': clustering.get(node, 0),
        'betweenness': betweenness.get(node, 0),
        'pagerank': pagerank.get(node, 0),
    })

df = pd.DataFrame(features)

# ==============================================================================
# STEP 4: Compute Z-scores (optional but useful for explainability)
# ==============================================================================

print("Computing z-scores...")

z_scores = (df.drop(columns=['node']) - df.drop(columns=['node']).mean()) / df.drop(columns=['node']).std()
z_scores.columns = [f"{col}_zscore" for col in z_scores.columns]
df = pd.concat([df, z_scores], axis=1)

# ==============================================================================
# STEP 5: Train Isolation Forest Model
# ==============================================================================

print("Running Isolation Forest...")

# Prepare input data (excluding node ID and z-scores)
X = df[[col for col in df.columns if not col.startswith('node') and not col.endswith('_zscore')]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Run Isolation Forest
model = IsolationForest(contamination=0.05, random_state=42)
df['anomaly_score'] = model.fit_predict(X_scaled)  # -1 = anomaly, 1 = normal

# ==============================================================================
# STEP 6: Split Results
# ==============================================================================

anomalous_nodes_df = df[df['anomaly_score'] == -1]
normal_nodes_df = df[df['anomaly_score'] == 1]

print(f"\nTotal nodes: {len(df)}")
print(f"Normal nodes: {len(normal_nodes_df)}")
print(f"Anomalous nodes: {len(anomalous_nodes_df)}")

# ==============================================================================
# STEP 7: Print Breakdown of Anomalous Nodes
# ==============================================================================

print("\nDetailed Feature and Z-Score Breakdown for Anomalous Nodes:")
cols_to_show = [
    'node', 'degree', 'clustering', 'betweenness', 'pagerank',
    'degree_zscore', 'clustering_zscore', 'betweenness_zscore', 'pagerank_zscore'
]
print(anomalous_nodes_df[cols_to_show].sort_values(by='pagerank', ascending=False).to_string(index=False))

# ==============================================================================
# STEP 8: Visualize the Graph with Anomalies Highlighted
# ==============================================================================

print("\nVisualizing anomalies...")

pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(14, 10))

# Draw edges
nx.draw_networkx_edges(G, pos, alpha=0.2)

# Draw nodes by group
nx.draw_networkx_nodes(G, pos,
    nodelist=normal_nodes_df['node'].tolist(),
    node_color='lightgray',
    node_size=40,
    label='Normal'
)
nx.draw_networkx_nodes(G, pos,
    nodelist=anomalous_nodes_df['node'].tolist(),
    node_color='red',
    node_size=80,
    label='Anomalous'
)

# Label only the anomalies
labels = {node: f"n{node}" for node in anomalous_nodes_df['node'].tolist()}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='black')

plt.title("Anomaly Detection (Undirected Network - Isolation Forest)", fontsize=14)
plt.legend(scatterpoints=1, loc='best')
plt.axis('off')
plt.tight_layout()
plt.show()
