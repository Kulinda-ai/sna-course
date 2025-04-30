# FILE: demo_louvain_method.py

import json
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain  # This is from the 'python-louvain' package
import pandas as pd

# ==============================================================================
# STEP 1: Load Graph from JSON
# ==============================================================================

def create_graph_from_json(nodes_file_path, edges_file_path):
    """
    Load nodes and edges from JSON into a NetworkX undirected graph.
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

# File paths
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Load the graph
G = create_graph_from_json(nodes_file_path, edges_file_path)

# ==============================================================================
# STEP 2: Run Louvain Community Detection
# ==============================================================================

# Louvain is a fast, greedy optimization method for modularity.
# It returns a partition: a dictionary mapping node → community ID
partition = community_louvain.best_partition(G)

# Group nodes by their community ID for display/export
community_to_nodes = {}
for node, community in partition.items():
    community_to_nodes.setdefault(community, []).append(node)

# Format output as a list of dictionaries for easy JSON use
communities_json_array = [
    {"community_id": community_id, "nodes_items": nodes}
    for community_id, nodes in community_to_nodes.items()
]

# Convert to formatted JSON and save
communities_json = json.dumps(communities_json_array, indent=4)
with open('communities.json', 'w') as file:
    file.write(communities_json)

# Print sample output
print(communities_json[:500])  # Print the first 500 characters for preview

# ==============================================================================
# STEP 3: Export to DataFrame for Analysis
# ==============================================================================

# Useful if you want to analyze the partition in tabular form
df_partition = pd.DataFrame(partition.items(), columns=['Node', 'Community'])
print(df_partition.head())

# ==============================================================================
# STEP 4: Visualize Detected Communities
# ==============================================================================

plt.figure(figsize=(10, 7))
pos = nx.spring_layout(G, seed=42)  # Stable layout for reproducibility

# Determine number of unique communities
num_communities = max(partition.values()) + 1
cmap = plt.cm.get_cmap('viridis', num_communities)

# Draw nodes colored by community assignment
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=list(partition.keys()),
    node_size=500,
    cmap=cmap,
    node_color=list(partition.values())
)
nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray')
nx.draw_networkx_labels(G, pos)

plt.title("Louvain Community Detection")
plt.axis('off')
plt.show()
