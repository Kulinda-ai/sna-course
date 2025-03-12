import json
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain  # Make sure it's python-louvain!
import pandas as pd

# Function to create the graph
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

# File paths
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Create the graph
G = create_graph_from_json(nodes_file_path, edges_file_path)

# Apply Louvain community detection
partition = community_louvain.best_partition(G)

# Group nodes by their community
community_to_nodes = {}
for node, community in partition.items():
    community_to_nodes.setdefault(community, []).append(node)

# Convert communities to JSON format
communities_json_array = [
    {"community_id": community_id, "nodes_items": nodes}
    for community_id, nodes in community_to_nodes.items()
]

# Convert to JSON string and optionally save
communities_json = json.dumps(communities_json_array, indent=4)

with open('communities.json', 'w') as file:
    file.write(communities_json)

# Print sample JSON
print(communities_json[:500])

# Optional: Export to pandas DataFrame for analysis
df_partition = pd.DataFrame(partition.items(), columns=['Node', 'Community'])
print(df_partition.head())

# Visualize the communities in the network
plt.figure(figsize=(10, 7))
pos = nx.spring_layout(G, seed=42)  # Stable layout with a seed

# Color nodes by community assignment
cmap = plt.cm.get_cmap('viridis', max(partition.values()) + 1)

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
