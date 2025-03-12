import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Create an empty graph
def create_graph_from_json(nodes_file_path, edges_file_path):
    # Load nodes
    with open(nodes_file_path, 'r') as file:
        nodes_data = json.load(file)
    
    # Load edges
    with open(edges_file_path, 'r') as file:
        edges_data = json.load(file)
    
    # Create a new graph
    G = nx.Graph()
    
    # Add nodes to the graph
    for node in nodes_data:
        G.add_node(node['id'], label=node.get('label', ''))
    
    # Add edges to the graph
    for edge in edges_data:
        G.add_edge(edge['source'], edge['target'])
    
    return G

# Replace 'networkx_nodes.json' and 'networkx_edges.json' with the actual paths to your files
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Create the graph
G = create_graph_from_json(nodes_file_path, edges_file_path)

eigenvector_centrality = nx.eigenvector_centrality(G)

# Print eigenvector centrality of each node
for node, centrality in eigenvector_centrality.items():
    print(f"{node}: {centrality}")

# Convert to DataFrame and Rank
df = pd.DataFrame(list(eigenvector_centrality.items()), columns=['Node', 'Eigenvector Centrality'])
df = df.sort_values(by='Eigenvector Centrality', ascending=False).reset_index(drop=True)
df.index += 1  # Set ranking to start from 1

# Print DataFrame
print(df)

# Save to JSON file
df.to_json("eigenvector_centrality.json", orient="records", indent=4)

print("\nSaved eigenvector centrality rankings to 'eigenvector_centrality.json'.")

