import json
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import greedy_modularity_communities

# === Load data and create the graph ===
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

# === Visualize the communities ===
def draw_communities(G, communities, title):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 7))
    
    # Map nodes to community index
    node_color_map = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_color_map[node] = idx

    colors = [node_color_map[node] for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.viridis, node_size=500, edgecolors='black')
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos)
    
    plt.title(title)
    plt.axis('off')
    plt.show()

# === Main Run ===
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Step 1: Load the graph
G = create_graph_from_json(nodes_file_path, edges_file_path)

# Step 2: Run NetworkX's built-in greedy modularity community detection (CNM algorithm)
communities = list(greedy_modularity_communities(G))

# Step 3: Print out the results
print(f"\n=== Greedy Modularity Communities (Clauset-Newman-Moore Algorithm) ===")
print(f"Total Communities Detected: {len(communities)}")
for idx, community in enumerate(communities, 1):
    print(f"Community {idx} ({len(community)} nodes): {sorted(community)}")

# Step 4: Calculate modularity (optional but useful)
modularity = nx.algorithms.community.quality.modularity(G, communities)
print(f"\nModularity of partition: {modularity:.4f}")

# Step 5: Draw communities
draw_communities(G, communities, "Greedy Modularity Communities (CNM Algorithm)")
