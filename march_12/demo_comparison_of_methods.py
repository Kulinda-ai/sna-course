# FILE: demo_comparison_of_methods.py

import json
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain  # External library for Louvain method
import time
from networkx.algorithms.community import (
    greedy_modularity_communities,
    label_propagation_communities,
    girvan_newman
)
import pandas as pd

# ==============================================================================
# STEP 1: Load Graph from JSON
# ==============================================================================

def create_graph_from_json(nodes_file_path, edges_file_path):
    """
    Creates an undirected graph from node and edge JSON files.
    Each node includes an 'id' and optionally a 'label'.
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

# Load your networkx-compatible data
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'
G = create_graph_from_json(nodes_file_path, edges_file_path)

# ==============================================================================
# STEP 2: Utility Functions
# ==============================================================================

def compute_modularity_from_communities(G, communities):
    """
    Converts a list of sets into a dict partition, then computes modularity.
    """
    partition = {}
    for cid, community in enumerate(communities):
        for node in community:
            partition[node] = cid
    return community_louvain.modularity(partition, G)

def draw_communities(G, communities, title):
    """
    Visualizes the given communities on a network layout.
    Each community is assigned a distinct color.
    """
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 7))

    node_color_map = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_color_map[node] = idx

    colors = [node_color_map[node] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=500, cmap=plt.cm.viridis, node_color=colors)
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray')
    nx.draw_networkx_labels(G, pos)
    plt.title(title)
    plt.axis('off')
    plt.show()

# ==============================================================================
# STEP 3: Run Community Detection Methods
# ==============================================================================

# -- A. Louvain Method --
# A popular method that greedily optimizes modularity. Often gives good results quickly.
start = time.time()
louvain_partition = community_louvain.best_partition(G)
end = time.time()

# Convert partition dictionary to list of sets
louvain_communities = {}
for node, cid in louvain_partition.items():
    louvain_communities.setdefault(cid, []).append(node)
louvain_communities = list(map(set, louvain_communities.values()))

louvain_modularity = community_louvain.modularity(louvain_partition, G)
louvain_time = end - start

# -- B. Greedy Modularity --
# A fast, modularity-based method built into NetworkX
start = time.time()
greedy_communities = list(greedy_modularity_communities(G))
end = time.time()
greedy_modularity = compute_modularity_from_communities(G, greedy_communities)
greedy_time = end - start

# -- C. Label Propagation --
# A fast, randomized algorithm that spreads labels until stable clusters form
start = time.time()
label_communities = list(label_propagation_communities(G))
end = time.time()
label_modularity = compute_modularity_from_communities(G, label_communities)
label_time = end - start

# -- D. Girvan-Newman --
# A hierarchical algorithm that removes edges with the highest betweenness
# We track modularity at each level to find the best split.
start = time.time()
gn_generator = girvan_newman(G)
gn_levels = []
gn_modularities = []

try:
    for level in range(5):  # Try 5 hierarchical splits
        communities = next(gn_generator)
        communities = list(map(set, communities))
        mod_score = compute_modularity_from_communities(G, communities)
        gn_levels.append(communities)
        gn_modularities.append(mod_score)
except StopIteration:
    print("Girvan-Newman completed before reaching 5 levels.")

end = time.time()

# Select best-performing GN level
if gn_levels:
    best_idx = gn_modularities.index(max(gn_modularities))
    best_gn_partition = gn_levels[best_idx]
    best_gn_modularity = gn_modularities[best_idx]
    best_gn_communities = best_gn_partition
else:
    best_idx = 0
    best_gn_modularity = 0
    best_gn_communities = [set(G.nodes())]

girvan_time = end - start

# ==============================================================================
# STEP 4: Summarize and Compare Results
# ==============================================================================

results = pd.DataFrame([
    {
        'Method': 'Louvain',
        'Communities': len(louvain_communities),
        'Modularity': round(louvain_modularity, 4),
        'Time (s)': round(louvain_time, 4)
    },
    {
        'Method': 'Greedy Modularity',
        'Communities': len(greedy_communities),
        'Modularity': round(greedy_modularity, 4),
        'Time (s)': round(greedy_time, 4)
    },
    {
        'Method': 'Label Propagation',
        'Communities': len(label_communities),
        'Modularity': round(label_modularity, 4),
        'Time (s)': round(label_time, 4)
    },
    {
        'Method': 'Girvan-Newman (Best of 5 levels)',
        'Communities': len(best_gn_communities),
        'Modularity': round(best_gn_modularity, 4),
        'Time (s)': round(girvan_time, 4)
    }
])

print("\n=== Community Detection Comparison ===")
print(results)

# ==============================================================================
# STEP 5: Visualize Detected Communities
# ==============================================================================

draw_communities(G, louvain_communities, "Louvain Communities")
draw_communities(G, greedy_communities, "Greedy Modularity Communities")
draw_communities(G, label_communities, "Label Propagation Communities")
draw_communities(G, best_gn_communities, "Girvan-Newman Communities (Best of 5 Levels)")
