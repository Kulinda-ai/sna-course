import json
import networkx as nx
import matplotlib.pyplot as plt

# === Load Data ===
with open("node_content.json", "r") as f:
    node_content = json.load(f)

with open("networkx_edges.json", "r") as f:
    edges = json.load(f)

# === Build Graph ===
G = nx.DiGraph()
for edge in edges:
    G.add_edge(str(edge["source"]), str(edge["target"]))

labels = {}
red_nodes = []

for node in node_content:
    node_id = str(node["id"])
    content = node.get("content", [])
    posts_count = len(content)
    attack_count = sum(1 for article in content if article.get("attack", False))
    percent_attack = round((attack_count / posts_count) * 100, 2) if posts_count else 0.0

    if percent_attack >= 75:
        label = "red"
        red_nodes.append({
            "id": node_id,
            "percent_attack": percent_attack,
            "posts_count": posts_count,
            "attack_count": attack_count
        })
    elif percent_attack <= 25:
        label = "green"
    else:
        label = "yellow"

    labels[node_id] = label
    G.nodes[node_id]["label"] = label

# === Print Top 10 Red Nodes ===
top_red = sorted(red_nodes, key=lambda x: x["percent_attack"], reverse=True)[:10]
print("\n🔥 Top 10 Red Nodes (≥ 75% attack content):")
for i, node in enumerate(top_red, 1):
    print(f"{i}. Node {node['id']} — {node['percent_attack']}% attack ({node['attack_count']}/{node['posts_count']} posts)")

# === Draw Network ===
pos = nx.spring_layout(G, seed=42)
colors = {"red": "red", "yellow": "gold", "green": "green"}
node_colors = [colors.get(G.nodes[n].get("label", "gray")) for n in G.nodes()]

plt.figure(figsize=(12, 9))
nx.draw_networkx_edges(G, pos, alpha=0.2)
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=150)
plt.title("Network with Nodes Classified by Attack % (Red ≥75%, Green ≤25%)")
plt.axis("off")
plt.tight_layout()
plt.show()
