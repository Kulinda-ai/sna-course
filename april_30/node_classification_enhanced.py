# FILE: node_classification_enhanced.py

import json
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import MinMaxScaler

# ==============================================================================
# STEP 1: Load Input Files
# ==============================================================================

with open("node_content.json") as f:
    node_content = json.load(f)

with open("networkx_nodes.json") as f:
    nodes_data = json.load(f)

with open("networkx_edges.json") as f:
    edges_data = json.load(f)

# ==============================================================================
# STEP 2: Build NetworkX Graph from Nodes and Edges
# ==============================================================================

G = nx.DiGraph()
for node in nodes_data:
    G.add_node(str(node["id"]))
for edge in edges_data:
    G.add_edge(str(edge["source"]), str(edge["target"]))

# ==============================================================================
# STEP 3: Compute Social Network Metrics (SNA)
# ==============================================================================

# We focus on three common metrics to represent node influence:
degree = dict(G.degree())
eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
betweenness = nx.betweenness_centrality(G)

# Combine into a DataFrame and normalize for scoring later
sna_df = pd.DataFrame({
    "id": list(degree.keys()),
    "degree": list(degree.values()),
    "eigenvector": [eigenvector.get(n, 0) for n in degree.keys()],
    "betweenness": [betweenness.get(n, 0) for n in degree.keys()]
}).set_index("id")

# Normalize to range [0, 1]
scaler = MinMaxScaler()
scaled = scaler.fit_transform(sna_df)
norm_df = pd.DataFrame(scaled, index=sna_df.index, columns=[f"{col}_norm" for col in sna_df.columns])
sna_df = sna_df.join(norm_df)

# ==============================================================================
# STEP 4: Analyze Risk Per Node (Combine ML + SNA)
# ==============================================================================

results = []

for node in node_content:
    node_id = str(node.get("id"))
    content = node.get("content", [])
    posts_count = len(content)

    # Skip nodes with no posts
    if posts_count == 0:
        continue

    # Determine % of posts marked as "attack"
    attack_count = sum(1 for article in content if article.get("attack", False))
    percent_attack = round((attack_count / posts_count) * 100, 2)

    # Color-code label based on thresholds
    if percent_attack >= 75:
        label = "red"
    elif percent_attack <= 25:
        label = "green"
    else:
        label = "yellow"

    # Get SNA metrics for this node
    sna = sna_df.loc[node_id] if node_id in sna_df.index else pd.Series({
        "degree": 0, "eigenvector": 0, "betweenness": 0,
        "degree_norm": 0, "eigenvector_norm": 0, "betweenness_norm": 0
    })

    # Composite risk score (tunable weights)
    risk_score = round(
        (percent_attack / 100) * 0.4 +
        (min(posts_count, 100) / 100) * 0.3 +
        sna["eigenvector_norm"] * 0.3,
        4
    )

    # Store node-level data
    results.append({
        "id": node_id,
        "label": label,
        "risk_score": risk_score,
        "percent_attack": percent_attack,
        "posts_count": posts_count,
        "attack_count": attack_count,
        "degree": sna["degree"],
        "eigenvector": sna["eigenvector"],
        "betweenness": sna["betweenness"],
        "degree_norm": sna["degree_norm"],
        "eigenvector_norm": sna["eigenvector_norm"],
        "betweenness_norm": sna["betweenness_norm"]
    })

# Save risk scoring output
with open("node_risk_scores.json", "w") as f:
    json.dump(results, f, indent=4)

print("✅ Scoring complete. Saved to 'node_risk_scores.json'")

# ==============================================================================
# STEP 5: Print Top 10 Risky Nodes
# ==============================================================================

top_10 = sorted(results, key=lambda x: x["risk_score"], reverse=True)[:10]
print("\n🔥 Top 10 Highest-Risk Nodes (excluding zero-post accounts):")
for i, node in enumerate(top_10, 1):
    print(f"{i}. Node {node['id']} — Risk Score: {node['risk_score']} — Attack %: {node['percent_attack']} — Posts: {node['posts_count']}")

# ==============================================================================
# STEP 6: Visualize the Risk Scoring on the Graph
# ==============================================================================

# Use only the subgraph of nodes with content
filtered_node_ids = set(node["id"] for node in results)
subgraph = G.subgraph(filtered_node_ids).copy()

# Attach risk scores to node attributes
for node in results:
    subgraph.nodes[node["id"]]["risk_score"] = node["risk_score"]

# Position and color configuration
pos = nx.spring_layout(subgraph, seed=42)
scores = [subgraph.nodes[n].get("risk_score", 0) for n in subgraph.nodes()]
norm = mcolors.Normalize(vmin=min(scores), vmax=max(scores))
cmap = plt.colormaps["Reds"]

plt.figure(figsize=(14, 10))
nx.draw_networkx_edges(subgraph, pos, alpha=0.2)

# Color nodes by risk
node_colors = [cmap(norm(subgraph.nodes[n]["risk_score"])) for n in subgraph.nodes()]
nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=150)

# Add node labels (IDs)
labels = {n: n for n in subgraph.nodes()}
nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=8)

# Add a colorbar to explain the score gradient
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array(scores)
plt.colorbar(sm, ax=plt.gca(), label="Composite Risk Score")

plt.title("Node Risk Visualization (Filtered + Eigenvector Centrality)", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.show()
