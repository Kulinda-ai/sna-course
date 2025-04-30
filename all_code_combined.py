# === annotate_files.py ===

# FILE: annotate_files.py

import os

def annotate_py_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                file_path = os.path.join(dirpath, filename)

                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # Check if first line already contains the filename
                header_comment = f"# FILE: {filename}"
                if lines and lines[0].strip() == header_comment:
                    continue  # Already annotated

                # Prepend filename comment and a blank line
                lines = [header_comment + '\n', '\n'] + lines

                # Write back to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)

                print(f"✅ Annotated: {file_path}")

# 🔧 Replace with your project root directory
project_root = "./"  # Current directory
annotate_py_files(project_root)


# === all_code_combined.py ===




# === create_master_file.py ===

import os

def create_master_code_file(root_dir, output_file="all_code_combined.py"):
    with open(output_file, "w", encoding="utf-8") as out:
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith(".py"):
                    file_path = os.path.join(dirpath, filename)
                    rel_path = os.path.relpath(file_path, root_dir)

                    out.write(f"# === {rel_path} ===\n\n")

                    with open(file_path, "r", encoding="utf-8") as f:
                        contents = f.read()

                    out.write(contents.strip() + "\n\n\n")

    print(f"✅ Combined code written to '{output_file}'")

# 🔧 Set to the root directory of your project
project_root = "./"  # or your actual folder path
create_master_code_file(project_root)


# === march_5/example_network_metrics.py ===

# FILE: example_network_metrics.py

import json
import networkx as nx
import pandas as pd
import numpy as np
import pandas as pd

def normalize_centrality(centrality_dict):
    """ Min-max normalization of centrality values """
    min_val = min(centrality_dict.values())
    max_val = max(centrality_dict.values())
    if max_val - min_val == 0:  # Avoid division by zero if all values are the same
        return {node: 0 for node in centrality_dict}
    return {node: (value - min_val) / (max_val - min_val) for node, value in centrality_dict.items()}

def analyze_network(G):
    # Table 1: Network Graph Overall Information
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)
    avg_shortest_path_length = np.mean([nx.average_shortest_path_length(G, weight='weight') if nx.is_connected(G) else float('nan')])
    avg_clustering = nx.average_clustering(G)
    global_betweenness = np.mean(list(nx.betweenness_centrality(G).values()))
    global_closeness = np.mean(list(nx.closeness_centrality(G).values()))

    network_info = pd.DataFrame({
        'Metric': ['Number of Nodes', 'Number of Edges', 'Density', 'Average Shortest Path Length', 
                   'Average Clustering', 'Global Betweenness Centrality', 'Global Closeness Centrality'],
        'Value': [num_nodes, num_edges, density, avg_shortest_path_length, 
                  avg_clustering, global_betweenness, global_closeness]
    })

    network_info_json = {
        'Number of Nodes': num_nodes,
        'Number of Edges': num_edges,
        'Density': density,
        'Average Shortest Path Length': avg_shortest_path_length, 
        'Average Clustering': avg_clustering, 
        'Global Betweenness Centrality': global_betweenness, 
        'Global Closeness Centrality': global_closeness,
    }
    
    # Table 2: Individual Node Metrics
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    degree_centrality = nx.degree_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-06)
    degrees = dict(G.degree())

    # Normalize betweenness and closeness centrality
    betweenness = normalize_centrality(betweenness)
    closeness = normalize_centrality(closeness)

    # Calculate eccentricity for each node
    if nx.is_connected(G):
        eccentricity = nx.eccentricity(G)
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
        eccentricity = nx.eccentricity(subgraph)
        for node in set(G.nodes()) - set(largest_cc):
            eccentricity[node] = None  # Placeholder for disconnected nodes
    
    # Customizable weights (sum must be 1)
    weights = {
        'degree': 0.4,
        'eigenvector': 0.3,
        'betweenness': 0.2,
        'closeness': 0.1,
    }

    # Ensure weights sum to 1
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    # Updated Influence Score Calculation
    influencer_scores = {
        node: (degree_centrality[node] * weights['degree'] +
               eigenvector_centrality[node] * weights['eigenvector'] +
               betweenness[node] * weights['betweenness'] +
               closeness[node] * weights['closeness'])
        for node in G.nodes()
    }
    
    nodes_info = pd.DataFrame({
        'Node': list(G.nodes()),
        'Betweenness': [betweenness[node] for node in G.nodes()],
        'Closeness': [closeness[node] for node in G.nodes()],
        'Eccentricity': [eccentricity[node] for node in G.nodes()],
        'Degree Centrality': [degree_centrality[node] for node in G.nodes()],
        'Eigenvector Centrality': [eigenvector_centrality[node] for node in G.nodes()],
        'Degrees': [degrees[node] for node in G.nodes()],
        'Influencer Score': [influencer_scores[node] for node in G.nodes()]
    })

    nodes_info['Rank'] = nodes_info['Influencer Score'].rank(ascending=False)
    
    # Sorting the DataFrame based on the influencer score
    nodes_info_sorted = nodes_info.sort_values(by='Influencer Score', ascending=False).reset_index(drop=True)
    
    # Exporting the DataFrame to JSON
    nodes_info_json = nodes_info_sorted.to_json(orient='records')
    
    # Save to a file
    with open('network_info.json', 'w') as f:
        json.dump(network_info_json, f, indent=4)

    with open('nodes_info.json', 'w') as f:
        json.dump(json.loads(nodes_info_json), f, indent=4)
    
    return network_info, nodes_info_sorted, network_info_json, nodes_info_json

# Example usage
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

network_info, nodes_info_sorted, network_info_json, nodes_info_json = analyze_network(G)

print("Network Information:")
print(network_info)
print("\nRanked Nodes Information:")
print(nodes_info_sorted)

# The JSON strings are also returned and saved to files, but here we'll just print a snippet to demonstrate
print("\nNetwork Information (JSON):")
print(json.dumps(network_info_json, indent=4)[:100] + '...')

print("\nNodes Information (JSON):")
print(nodes_info_json[:100] + '...')


# === march_5/example_predicted_connections.py ===

# FILE: example_predicted_connections.py

import json
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
import pandas as pd

# Create an empty graph
G = nx.Graph()

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

def predict_connections_for_cliques(graph):
    predictions = []
    for node in graph.nodes():
        neighbors = set(nx.neighbors(graph, node))
        for non_neighbor in set(graph.nodes()) - neighbors - {node}:
            common_neighbors = set(nx.common_neighbors(graph, node, non_neighbor))
            # We only consider pairs with at least two common neighbors to form a potential 3-node clique
            if len(common_neighbors) >= 2:
                prediction = {
                    "node": node,
                    "connected_node": non_neighbor,
                    'common_neighbors_count': len(common_neighbors),
                    "common_neighbors": list(common_neighbors)
                }
                predictions.append(prediction)
    return predictions

# Use the updated function
potential_connections_for_cliques = predict_connections_for_cliques(G)

# Convert to JSON string
connections_for_cliques_json = json.dumps(potential_connections_for_cliques, indent=4)

# Optionally, save to a file
with open('predicted_current_connections.json', 'w') as file:
    file.write(connections_for_cliques_json)

# Print part of the JSON string for demonstration
print(connections_for_cliques_json[:500])


# === march_5/example_eigenvector_centrality.py ===

# FILE: example_eigenvector_centrality.py

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


# === march_5/example_closeness_centrality.py ===

# FILE: example_closeness_centrality.py

import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

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

# Calculate closeness centrality
closeness_centrality = nx.closeness_centrality(G)

# Print closeness centrality of each node
for node, centrality in closeness_centrality.items():
    print(f"{node}: {centrality}")

# Convert to DataFrame and Rank
df = pd.DataFrame(list(closeness_centrality.items()), columns=['Node', 'Closeness Centrality'])
df = df.sort_values(by='Closeness Centrality', ascending=False).reset_index(drop=True)
df.index += 1  # Set ranking to start from 1

# Print DataFrame
print(df)

# Save to JSON file
df.to_json("closeness_centrality.json", orient="records", indent=4)

print("\nSaved closeness centrality rankings to 'closeness_centrality.json'.")


# === march_5/example_degree_connections.py ===

# FILE: example_degree_connections.py

import networkx as nx
import json
import pandas as pd

def find_connections_by_degree(G):
    connections_by_degree = {}
    
    for node in G.nodes():
        # Initialize dictionaries for each node
        connections_by_degree[node] = {'1st': [], '2nd': [], '3rd': []}
        
        # 1st degree connections
        first_degree = set(G.neighbors(node))
        connections_by_degree[node]['1st'] = list(first_degree)
        
        # 2nd degree connections
        second_degree = set()
        for neighbor in first_degree:
            second_degree.update(G.neighbors(neighbor))
        # Remove the original node and already counted 1st degree connections
        second_degree = second_degree - first_degree - {node}
        connections_by_degree[node]['2nd'] = list(second_degree)
        
        # 3rd degree connections
        third_degree = set()
        for neighbor in second_degree:
            third_degree.update(G.neighbors(neighbor))
        # Remove nodes already counted in 1st and 2nd degree connections, and the original node
        third_degree = third_degree - first_degree - second_degree - {node}
        connections_by_degree[node]['3rd'] = list(third_degree)
    
    return connections_by_degree
    
def connections_to_json(G):
    connections_by_degree = find_connections_by_degree(G)
    # Convert the connections data to a JSON string
    json_data = json.dumps(connections_by_degree, indent=4)
    return json_data

# Example usage
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

json_data = connections_to_json(G)
print(json_data)

# Optionally, write the JSON data to a file
with open('network_connections_by_degree.json', 'w') as f:
    json.dump(json.loads(json_data), f, indent=4)


# === march_5/example_cliques_detection.py ===

# FILE: example_cliques_detection.py

import json
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
import pandas as pd

# Create an empty graph
G = nx.Graph()

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

# Find all maximal cliques in the graph
cliques = list(nx.find_cliques(G))

# Filter cliques to include only those with 3 or more nodes
filtered_cliques = [clique for clique in cliques if len(clique) >= 3]

# Convert filtered cliques into the specified JSON format
filtered_cliques_json_array = [
    {"clique_id": i, "nodes": clique}
    for i, clique in enumerate(filtered_cliques)
]

# Convert to JSON string
filtered_cliques_json = json.dumps(filtered_cliques_json_array, indent=4)

# Optionally, save to a file
with open('cliques.json', 'w') as file:
    file.write(filtered_cliques_json)

# Print the JSON string
print(filtered_cliques_json[:500])


# === march_5/example_at_risk_identification.py ===

# FILE: example_at_risk_identification.py

import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Create a weighted graph
def create_graph_from_json(nodes_file_path, edges_file_path):
    with open(nodes_file_path, 'r') as file:
        nodes_data = json.load(file)
    with open(edges_file_path, 'r') as file:
        edges_data = json.load(file)

    G = nx.Graph()

    for node in nodes_data:
        G.add_node(node['id'], label=node.get('label', ''))

    for edge in edges_data:
        weight = edge.get('weight', 1)
        G.add_edge(edge['source'], edge['target'], weight=weight)

    return G

# Load the graph
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'
G = create_graph_from_json(nodes_file_path, edges_file_path)

# Step 2: Calculate centrality measures
degree_centrality = nx.degree_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)

# Step 3: Normalize centrality measures
def normalize_centrality(centrality_dict):
    min_val = min(centrality_dict.values())
    max_val = max(centrality_dict.values())
    if max_val - min_val == 0:
        return {node: 0 for node in centrality_dict}
    return {node: (value - min_val) / (max_val - min_val) for node, value in centrality_dict.items()}

betweenness_centrality = normalize_centrality(betweenness_centrality)
closeness_centrality = normalize_centrality(closeness_centrality)

# Step 4: Influence Score Calculation
weights = {'degree': 0.4, 'eigenvector': 0.3, 'betweenness': 0.2, 'closeness': 0.1}
total_weight = sum(weights.values())
weights = {k: v / total_weight for k, v in weights.items()}  # Ensure sum = 1

def influence_score(node):
    return (degree_centrality[node] * weights['degree'] +
            eigenvector_centrality[node] * weights['eigenvector'] +
            betweenness_centrality[node] * weights['betweenness'] +
            closeness_centrality[node] * weights['closeness'])

# Compute influence scores for all nodes
influence_scores = {node: influence_score(node) for node in G.nodes()}

# Step 5: Identify Influencers (Top 10%)
num_influencers = max(1, int(len(G.nodes()) * 0.1))
sorted_nodes = sorted(influence_scores, key=influence_scores.get, reverse=True)
influencers = set(sorted_nodes[:num_influencers])

print(f"Top {num_influencers} Influencers Identified:")
for rank, node in enumerate(sorted_nodes[:num_influencers], start=1):
    print(f"{rank}. {node} (Score: {influence_scores[node]:.4f})")

# Step 6: Identify At-Risk Accounts in Two Categories
mildly_at_risk = []   # Connected to exactly 1 influencer
highly_at_risk = []   # Connected to 2 or more influencers

for node in G.nodes():
    if node in influencers:
        continue  # Skip influencers
    
    neighbors = set(G.neighbors(node))
    influencer_neighbors = neighbors.intersection(influencers)

    if len(influencer_neighbors) == 1:
        mildly_at_risk.append(node)
    elif len(influencer_neighbors) > 1:
        highly_at_risk.append(node)

# Step 7: Print At-Risk Accounts
print("\nMildly At-Risk Accounts (Connected to 1 Influencer):")
for node in mildly_at_risk:
    print(f"Node {node}: Connected to 1 Influencer")

print("\nHighly At-Risk Accounts (Connected to Multiple Influencers):")
for node in highly_at_risk:
    print(f"Node {node}: Connected to {len(set(G.neighbors(node)).intersection(influencers))} Influencers")

# Step 8: Visualization
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.3, iterations=50)

# Color nodes based on categories
node_colors = []
for node in G.nodes():
    if node in influencers:
        node_colors.append('red')  # Influencers
    elif node in mildly_at_risk:
        node_colors.append('orange')  # Mildly at risk
    elif node in highly_at_risk:
        node_colors.append('purple')  # Highly at risk
    else:
        node_colors.append('skyblue')  # Other nodes

node_sizes = [influence_scores[node] * 1000 for node in G.nodes()]

nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color=node_colors, edge_color='gray', font_size=8)

# Label categories
plt.title("Graph Visualization - Red = Influencers, Orange = Mildly At-Risk, Purple = Highly At-Risk")
plt.show()


# === march_5/example_degree_centrality.py ===

# FILE: example_degree_centrality.py

import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Create an empty graph
G = nx.Graph()

# Add nodes
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

# Calculate degree centrality
degree_centrality = nx.degree_centrality(G)

# Print degree centrality of each node
for node, centrality in degree_centrality.items():
    print(f"{node}: {centrality}")

# Convert to DataFrame and Rank
df = pd.DataFrame(list(degree_centrality.items()), columns=['Node', 'Degree Centrality'])
df = df.sort_values(by='Degree Centrality', ascending=False).reset_index(drop=True)
df.index += 1  # Set ranking to start from 1

# Print DataFrame
print(df)

# Save to JSON file
df.to_json("degree_centrality.json", orient="records", indent=4)

print("\nSaved degree centrality rankings to 'degree_centrality.json'.")


# === march_5/example_influencer_ranking_normalized.py ===

# FILE: example_influencer_ranking_normalized.py

import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Step 1: Create a weighted graph
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
    
    # Add edges with weights if available
    for edge in edges_data:
        weight = edge.get('weight', 1)  # Default weight to 1 if not specified
        G.add_edge(edge['source'], edge['target'], weight=weight)
    
    return G

# Replace 'networkx_nodes.json' and 'networkx_edges.json' with the actual paths
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Create the graph
G = create_graph_from_json(nodes_file_path, edges_file_path)

# Step 2: Calculate centrality measures
degree_centrality = nx.degree_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)

# Step 3: Normalize centrality measures that need it
def normalize_centrality(centrality_dict):
    """ Min-max normalization of centrality values """
    min_val = min(centrality_dict.values())
    max_val = max(centrality_dict.values())
    if max_val - min_val == 0:  # Avoid division by zero if all values are the same
        return {node: 0 for node in centrality_dict}
    return {node: (value - min_val) / (max_val - min_val) for node, value in centrality_dict.items()}

# Normalize betweenness and closeness centrality
betweenness_centrality = normalize_centrality(betweenness_centrality)
closeness_centrality = normalize_centrality(closeness_centrality)

# Step 4: Analyze connection quality (average weight)
def average_weight(G, node):
    """ Calculate the average weight of edges connected to a node """
    total_weight = sum(weight for _, _, weight in G.edges(node, data='weight'))
    return total_weight / G.degree(node) if G.degree(node) > 0 else 0

# Step 5: Influence Score Calculation
# Customizable weights for each centrality metric (must sum to 1)
weights = {
    'degree': 0.4,
    'eigenvector': 0.3,
    'betweenness': 0.2,
    'closeness': 0.1,
}

# Auto-normalize weights if necessary
total_weight = sum(weights.values())
weights = {k: v / total_weight for k, v in weights.items()}  # Ensure sum = 1

def influence_score(node):
    """ Compute influence score as weighted sum of normalized centrality measures """
    return (degree_centrality[node] * weights['degree'] +
            eigenvector_centrality[node] * weights['eigenvector'] +
            betweenness_centrality[node] * weights['betweenness'] +
            closeness_centrality[node] * weights['closeness'])

# Compute influence scores for all nodes
influence_scores = {node: influence_score(node) for node in G.nodes()}

# Step 6: Rank nodes from most to least influential
ranked_nodes = sorted(influence_scores, key=influence_scores.get, reverse=True)

# Print ranked list of nodes
print("Nodes ranked from most to least influential:")
for rank, node in enumerate(ranked_nodes, start=1):
    print(f"{rank}. {node} (Score: {influence_scores[node]:.4f})")

# Step 7: Visualization (Optional)
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G)  # Layout for visualization
node_sizes = [influence_scores[node] * 1000 for node in G.nodes()]  # Scale sizes

nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color='skyblue', edge_color='gray', font_size=10)
plt.title("Graph with Node Influence Scaling")
plt.show()


# === march_5/example_betweenness_centrality.py ===

# FILE: example_betweenness_centrality.py

import json
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

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

# Calculate betweenness centrality
betweenness_centrality = nx.betweenness_centrality(G)

# Print betweenness centrality of each node
for node, centrality in betweenness_centrality.items():
    print(f"{node}: {centrality}")

# Convert to DataFrame and Rank
df = pd.DataFrame(list(betweenness_centrality.items()), columns=['Node', 'Betweenness Centrality'])
df = df.sort_values(by='Betweenness Centrality', ascending=False).reset_index(drop=True)
df.index += 1  # Set ranking to start from 1

# Print DataFrame
print(df)

# Save to JSON file
df.to_json("betweenness_centrality.json", orient="records", indent=4)

print("\nSaved betweenness centrality rankings to 'betweenness_centrality.json'.")


# === march_5/example_community_types.py ===

# FILE: example_community_types.py

import json
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import (
    greedy_modularity_communities,
    label_propagation_communities,
    girvan_newman
)

# Load data and create the graph
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

# Replace with your actual file paths
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Create graph
G = create_graph_from_json(nodes_file_path, edges_file_path)

# Helper function to visualize communities
def draw_communities(G, communities, title):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 7))
    
    # Flatten communities into a node -> community index mapping
    node_color_map = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_color_map[node] = idx

    colors = [node_color_map[node] for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.viridis, node_size=500)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos)
    plt.title(title)
    plt.axis('off')
    plt.show()

# 1. Greedy Modularity Communities
greedy_communities = list(greedy_modularity_communities(G))
print(f"Greedy Modularity detected {len(greedy_communities)} communities.")
draw_communities(G, greedy_communities, "Greedy Modularity Communities")

# 2. Label Propagation Communities
label_prop_communities = list(label_propagation_communities(G))
print(f"Label Propagation detected {len(label_prop_communities)} communities.")
draw_communities(G, label_prop_communities, "Label Propagation Communities")

# 3. Girvan-Newman Communities
# Get first level of communities (split into 2 groups)
girvan_newman_generator = girvan_newman(G)
first_level_communities = next(girvan_newman_generator)  # First split
print(f"Girvan-Newman (first split) detected {len(first_level_communities)} communities.")
draw_communities(G, first_level_communities, "Girvan-Newman Communities (First Split)")

# Optional: Get next split if you want to go deeper
try:
    second_level_communities = next(girvan_newman_generator)
    print(f"Girvan-Newman (second split) detected {len(second_level_communities)} communities.")
    draw_communities(G, second_level_communities, "Girvan-Newman Communities (Second Split)")
except StopIteration:
    print("No more splits available.")


# === april_16/centrality_comparison.py ===

# FILE: centrality_comparison.py

import json
import networkx as nx
import csv

# === Load raw data ===
with open('networkx.json', 'r') as f:
    raw_data = json.load(f)

# Dust filter threshold
dust_threshold = 0.00001

# === Helper: Build graph from raw JSON ===
def build_graph(use_weights=False):
    G = nx.DiGraph()

    # Add all nodes
    for node in raw_data['nodes']:
        G.add_node(node['data']['id'], **node['data'])

    # Add edges
    for edge in raw_data['edges']:
        edge_data = edge['data']
        source = edge_data['source']
        target = edge_data['target']
        value = edge_data.get('value', 0)

        if edge_data.get('dust') is True or value <= dust_threshold:
            continue  # Skip dust

        attrs = {'value': value}
        if use_weights:
            attrs['weight'] = value

        G.add_edge(source, target, **attrs)

    # Remove nodes with no edges
    nodes_with_edges = set()
    for u, v in G.edges():
        nodes_with_edges.add(u)
        nodes_with_edges.add(v)

    for node in list(G.nodes):
        if node not in nodes_with_edges:
            G.remove_node(node)

    return G

# === Build both graphs ===
G_unweighted = build_graph(use_weights=False)
G_weighted = build_graph(use_weights=True)

print(f"Unweighted: {G_unweighted.number_of_nodes()} nodes, {G_unweighted.number_of_edges()} edges")
print(f"Weighted:   {G_weighted.number_of_nodes()} nodes, {G_weighted.number_of_edges()} edges")

# === Compute centralities ===

# Unweighted
pagerank_un = nx.pagerank(G_unweighted)
betweenness_un = nx.betweenness_centrality(G_unweighted)

# Weighted
pagerank_wt = nx.pagerank(G_weighted, weight='weight')
betweenness_wt = nx.betweenness_centrality(G_weighted, weight='weight')

# === Join & Export ===

all_ids = set(G_unweighted.nodes()).union(set(G_weighted.nodes()))
rows = []

for node_id in all_ids:
    row = {
        'account_id': node_id,
        'wallet_type': G_unweighted.nodes[node_id].get('wallet_type', G_weighted.nodes[node_id].get('wallet_type')),
        'pagerank_unweighted': pagerank_un.get(node_id, 0),
        'pagerank_weighted': pagerank_wt.get(node_id, 0),
        'betweenness_unweighted': betweenness_un.get(node_id, 0),
        'betweenness_weighted': betweenness_wt.get(node_id, 0),
    }
    rows.append(row)

# Write to CSV
with open('centrality_comparison.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'account_id',
        'wallet_type',
        'pagerank_unweighted', 'pagerank_weighted',
        'betweenness_unweighted', 'betweenness_weighted'
    ])
    writer.writeheader()
    writer.writerows(rows)

print("✅ Exported centrality_comparison.csv with both weighted and unweighted metrics.")


# === april_16/app.py ===

# FILE: app.py

from flask import Flask, render_template, jsonify
import json

app = Flask(__name__)

# === Route: Initial Analysis ===
@app.route('/initial-analysis/')
def initial_analysis_view():
    return render_template('initial-analysis.html')

@app.route('/data-initial-analysis')
def initial_analysis_data():
    with open('initial_analysis.json', 'r') as f:
        graph_data = json.load(f)
    return jsonify(graph_data)

if __name__ == '__main__':
    app.run(debug=True)


# === april_16/initial_analysis.py ===

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


# === april_30/eon_visualization_influential_node.py ===

# FILE: eon_visualization_influential_node.py

import json
import networkx as nx
import EoN
import matplotlib.pyplot as plt
from collections import Counter

# === Load Nodes and Edges ===
with open('networkx_nodes.json') as f:
    node_data = json.load(f)

with open('networkx_edges.json') as f:
    edge_data = json.load(f)

# === Create Graph ===
G = nx.DiGraph()
for node in node_data:
    G.add_node(str(node['id']), label=node['label'])

for edge in edge_data:
    G.add_edge(str(edge['source']), str(edge['target']))

# === Parameters ===
tau = 0.2        # Transmission rate
gamma = 0.05     # Recovery rate
initial_infecteds = ["191"]

# === Run Simulation ===
sim = EoN.fast_SIR(G, tau=tau, gamma=gamma,
                   initial_infecteds=initial_infecteds,
                   return_full_data=True)

# === Layout for Drawing (same for both graphs)
pos = nx.spring_layout(G, seed=42)

# === Function to visualize status at a given time ===
def visualize_status_at_time(time_label, status_dict):
    sus_nodes = [n for n in G.nodes if status_dict.get(n, 'S') == 'S']
    inf_nodes = [n for n in G.nodes if status_dict.get(n, 'S') == 'I']
    rec_nodes = [n for n in G.nodes if status_dict.get(n, 'S') == 'R']

    print(f"\n--- {time_label} Status Breakdown ---")
    print("Susceptible:", len(sus_nodes), "| Infected:", len(inf_nodes), "| Recovered:", len(rec_nodes))
    print("Sample Infected Nodes:", inf_nodes[:10])
    print("Sample Recovered Nodes:", rec_nodes[:10])

    plt.figure(figsize=(14, 10))
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    nx.draw_networkx_nodes(G, pos, nodelist=sus_nodes, node_color='blue', node_size=40, label='Susceptible')
    nx.draw_networkx_nodes(G, pos, nodelist=inf_nodes, node_color='orange', node_size=80, label='Infected')
    nx.draw_networkx_nodes(G, pos, nodelist=rec_nodes, node_color='green', node_size=80, label='Recovered')
    plt.title(f"Information Spread - {time_label}", fontsize=14)
    plt.legend(scatterpoints=1, loc='best')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# === Visualize at Midpoint ===
mid_idx = round(len(sim.t()) / 2)
mid_time = sim.t()[mid_idx]
mid_status = sim.get_statuses(time=mid_time)
visualize_status_at_time(f"Midpoint (t = {mid_time:.2f})", mid_status)

# === Visualize at Final State ===
final_time = sim.t()[-1]
final_status = sim.get_statuses(time=final_time)
visualize_status_at_time(f"Final State (t = {final_time:.2f})", final_status)

# === Summary Info ===
print("\n=== SIMULATION SUMMARY ===")
print(f"Initial infected node(s): {initial_infecteds}")
print(f"Total nodes in graph: {G.number_of_nodes()}")
print(f"Time steps in simulation: {len(sim.t())}")
print(f"Final susceptible count: {sim.S()[-1]}")
print(f"Final infected count: {sim.I()[-1]}")
print(f"Final recovered count: {sim.R()[-1]}")
print("=================================")


# === april_30/node_classification.py ===

# FILE: node_classification.py

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


# === april_30/eon_example_top_influential_nodes_rapid_spread.py ===

# FILE: eon_example_top_influential_nodes_rapid_spread.py

import json
import networkx as nx
import EoN
import matplotlib.pyplot as plt

# === Load Nodes and Edges ===
print("Loading nodes and edges...")
with open('networkx_nodes.json') as f:
    node_data = json.load(f)

with open('networkx_edges.json') as f:
    edge_data = json.load(f)

# === Create a Directed Graph ===
G = nx.DiGraph()
print("Creating graph...")

for node in node_data:
    G.add_node(str(node['id']), label=node.get('label', ''))

for edge in edge_data:
    G.add_edge(str(edge['source']), str(edge['target']))

print(f"Total nodes: {G.number_of_nodes()}, Total edges: {G.number_of_edges()}")

# === Calculate Eigenvector Centrality ===
print("Calculating eigenvector centrality...")
try:
    centrality = nx.eigenvector_centrality_numpy(G)
except:
    centrality = nx.eigenvector_centrality_numpy(G.to_undirected())  # fallback if directed fails

# === Get Top 5 Nodes ===
top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
initial_infecteds = [node for node, _ in top_nodes]
print(f"Top 5 influential nodes by eigenvector centrality: {initial_infecteds}")

# === Define SIR Parameters ===
tau = 0.8
gamma = 0.05
print(f"Running SIR simulation with tau={tau}, gamma={gamma}...")

# === Run Simulation ===
t, S, I, R = EoN.fast_SIR(G, tau=tau, gamma=gamma, initial_infecteds=initial_infecteds)

# === Output Final Stats ===
print("Simulation finished.")
print(f"  Final susceptible: {S[-1]}")
print(f"  Final infected: {I[-1]}")
print(f"  Final recovered: {R[-1]}")
print(f"  Total simulation time: {t[-1]:.2f}")

# === Plot the Results ===
plt.figure(figsize=(10, 6))
plt.plot(t, S, label="Susceptible", color='blue')
plt.plot(t, I, label="Infected", color='orange')
plt.plot(t, R, label="Recovered", color='green')
plt.xlabel("Time")
plt.ylabel("Number of Nodes")
plt.title("Information Spread Simulation (SIR Model)\nTop 5 Influential Nodes as Sources w/ Rapid Spread")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Done.")


# === april_30/eon_example_low_influence_node_rapid_spread.py ===

# FILE: eon_example_low_influence_node_rapid_spread.py

import json
import networkx as nx
import EoN
import matplotlib.pyplot as plt

# === Load Nodes and Edges ===
print("Loading nodes and edges...")

with open('networkx_nodes.json') as f:
    node_data = json.load(f)

with open('networkx_edges.json') as f:
    edge_data = json.load(f)

# === Create a Directed Graph ===
G = nx.DiGraph()
print(f"Creating graph...")

# Add nodes (use string IDs to match edge references)
for node in node_data:
    node_id = str(node['id'])
    G.add_node(node_id, label=node.get('label', ''))
print(f"Total nodes added: {G.number_of_nodes()}")

# Add edges (string IDs to match node IDs)
for edge in edge_data:
    source = str(edge['source'])
    target = str(edge['target'])
    G.add_edge(source, target)
print(f"Total edges added: {G.number_of_edges()}")

# === Define SIR Model Parameters ===
tau = 0.8    # Transmission rate
gamma = 0.05 # Recovery rate
print(f"Model parameters set: tau = {tau}, gamma = {gamma}")

# === Choose Initial Infected Node(s) ===
# Use a node with low centrality and low degree
initial_infecteds = ["103"]
print(f"Initial infected node(s): {initial_infecteds}")

# === Run the SIR Simulation ===
print("Running SIR simulation using EoN...")
t, S, I, R = EoN.fast_SIR(G, tau=tau, gamma=gamma, initial_infecteds=initial_infecteds)

# === Display Final Results ===
print(f"Simulation finished. Final statistics:")
print(f"  Final susceptible: {S[-1]}")
print(f"  Final infected: {I[-1]}")
print(f"  Final recovered: {R[-1]}")
print(f"  Total simulation time: {t[-1]:.2f}")

# === Plot the Results ===
print("Plotting the SIR spread over time...")
plt.figure(figsize=(10, 6))
plt.plot(t, S, label="Susceptible", color='blue')
plt.plot(t, I, label="Infected", color='orange')
plt.plot(t, R, label="Recovered", color='green')
plt.xlabel("Time")
plt.ylabel("Number of Nodes")
plt.title("Information Spread Simulation (SIR Model) - Low-Influence Node (Node 103) w/ Rapid Spread")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Done.")


# === april_30/anomaly_detection_undirected_graph.py ===

# FILE: anomaly_detection_undirected_graph.py

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

# === Fix Labels (optional)
for node in node_data:
    node['label'] = f"node_{node['id']}"

# === Create Undirected Graph ===
G = nx.Graph()
for node in node_data:
    G.add_node(str(node['id']), label=node['label'])

for edge in edge_data:
    G.add_edge(str(edge['source']), str(edge['target']))

# === Compute Node Features ===
print("Computing network features...")
betweenness = nx.betweenness_centrality(G)
pagerank = nx.pagerank(G)  # works on undirected graphs too
clustering = nx.clustering(G)

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

# === Compute Z-scores ===
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
cols_to_show = ['node', 'degree', 'clustering', 'betweenness', 'pagerank',
                'degree_zscore', 'clustering_zscore', 'betweenness_zscore', 'pagerank_zscore']
print(anomalous_nodes_df[cols_to_show].sort_values(by='pagerank', ascending=False).to_string(index=False))

# === Visualize the Graph with Labels ===
print("\nVisualizing anomalies...")
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(14, 10))
nx.draw_networkx_edges(G, pos, alpha=0.2)
nx.draw_networkx_nodes(G, pos, nodelist=normal_nodes_df['node'].tolist(), node_color='lightgray', node_size=40, label='Normal')
nx.draw_networkx_nodes(G, pos, nodelist=anomalous_nodes_df['node'].tolist(), node_color='red', node_size=80, label='Anomalous')

# Add labels to anomalous nodes
labels = {node: f"n{node}" for node in anomalous_nodes_df['node'].tolist()}
nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='black')

plt.title("Anomaly Detection (Undirected Network - Isolation Forest)", fontsize=14)
plt.legend(scatterpoints=1, loc='best')
plt.axis('off')
plt.tight_layout()
plt.show()


# === april_30/eon_example_low_influence_node.py ===

# FILE: eon_example_low_influence_node.py

import json
import networkx as nx
import EoN
import matplotlib.pyplot as plt

# === Load Nodes and Edges ===
print("Loading nodes and edges...")

with open('networkx_nodes.json') as f:
    node_data = json.load(f)

with open('networkx_edges.json') as f:
    edge_data = json.load(f)

# === Create a Directed Graph ===
G = nx.DiGraph()
print(f"Creating graph...")

# Add nodes (use string IDs to match edge references)
for node in node_data:
    node_id = str(node['id'])
    G.add_node(node_id, label=node.get('label', ''))
print(f"Total nodes added: {G.number_of_nodes()}")

# Add edges (string IDs to match node IDs)
for edge in edge_data:
    source = str(edge['source'])
    target = str(edge['target'])
    G.add_edge(source, target)
print(f"Total edges added: {G.number_of_edges()}")

# === Define SIR Model Parameters ===
tau = 0.2    # Transmission rate
gamma = 0.05 # Recovery rate
print(f"Model parameters set: tau = {tau}, gamma = {gamma}")

# === Choose Initial Infected Node(s) ===
# Use a node with low centrality and low degree
initial_infecteds = ["103"]
print(f"Initial infected node(s): {initial_infecteds}")

# === Run the SIR Simulation ===
print("Running SIR simulation using EoN...")
t, S, I, R = EoN.fast_SIR(G, tau=tau, gamma=gamma, initial_infecteds=initial_infecteds)

# === Display Final Results ===
print(f"Simulation finished. Final statistics:")
print(f"  Final susceptible: {S[-1]}")
print(f"  Final infected: {I[-1]}")
print(f"  Final recovered: {R[-1]}")
print(f"  Total simulation time: {t[-1]:.2f}")

# === Plot the Results ===
print("Plotting the SIR spread over time...")
plt.figure(figsize=(10, 6))
plt.plot(t, S, label="Susceptible", color='blue')
plt.plot(t, I, label="Infected", color='orange')
plt.plot(t, R, label="Recovered", color='green')
plt.xlabel("Time")
plt.ylabel("Number of Nodes")
plt.title("Information Spread Simulation (SIR Model) - Low-Influence Node (Node 103)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Done.")


# === april_30/eon_example_top_influential_nodes.py ===

# FILE: eon_example_top_influential_nodes.py

import json
import networkx as nx
import EoN
import matplotlib.pyplot as plt

# === Load Nodes and Edges ===
print("Loading nodes and edges...")
with open('networkx_nodes.json') as f:
    node_data = json.load(f)

with open('networkx_edges.json') as f:
    edge_data = json.load(f)

# === Create a Directed Graph ===
G = nx.DiGraph()
print("Creating graph...")

for node in node_data:
    G.add_node(str(node['id']), label=node.get('label', ''))

for edge in edge_data:
    G.add_edge(str(edge['source']), str(edge['target']))

print(f"Total nodes: {G.number_of_nodes()}, Total edges: {G.number_of_edges()}")

# === Calculate Eigenvector Centrality ===
print("Calculating eigenvector centrality...")
try:
    centrality = nx.eigenvector_centrality_numpy(G)
except:
    centrality = nx.eigenvector_centrality_numpy(G.to_undirected())  # fallback if directed fails

# === Get Top 5 Nodes ===
top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
initial_infecteds = [node for node, _ in top_nodes]
print(f"Top 5 influential nodes by eigenvector centrality: {initial_infecteds}")

# === Define SIR Parameters ===
tau = 0.2
gamma = 0.05
print(f"Running SIR simulation with tau={tau}, gamma={gamma}...")

# === Run Simulation ===
t, S, I, R = EoN.fast_SIR(G, tau=tau, gamma=gamma, initial_infecteds=initial_infecteds)

# === Output Final Stats ===
print("Simulation finished.")
print(f"  Final susceptible: {S[-1]}")
print(f"  Final infected: {I[-1]}")
print(f"  Final recovered: {R[-1]}")
print(f"  Total simulation time: {t[-1]:.2f}")

# === Plot the Results ===
plt.figure(figsize=(10, 6))
plt.plot(t, S, label="Susceptible", color='blue')
plt.plot(t, I, label="Infected", color='orange')
plt.plot(t, R, label="Recovered", color='green')
plt.xlabel("Time")
plt.ylabel("Number of Nodes")
plt.title("Information Spread Simulation (SIR Model)\nTop 5 Influential Nodes as Sources")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Done.")


# === april_30/eon_example_random_node.py ===

# FILE: eon_example_random_node.py

# 

import json
import networkx as nx
import EoN
import matplotlib.pyplot as plt

# === Load Nodes and Edges ===
print("Loading nodes and edges...")

with open('networkx_nodes.json') as f:
    node_data = json.load(f)

with open('networkx_edges.json') as f:
    edge_data = json.load(f)

# === Create a Directed Graph ===
G = nx.DiGraph()
print(f"Creating graph...")

# Add nodes (use string IDs to match edge references)
for node in node_data:
    node_id = str(node['id'])
    G.add_node(node_id, label=node.get('label', ''))
print(f"Total nodes added: {G.number_of_nodes()}")

# Add edges (string IDs to match node IDs)
for edge in edge_data:
    source = str(edge['source'])
    target = str(edge['target'])
    G.add_edge(source, target)
print(f"Total edges added: {G.number_of_edges()}")

# === Define SIR Model Parameters ===
tau = 0.2    # Transmission rate
gamma = 0.05 # Recovery rate
print(f"Model parameters set: tau = {tau}, gamma = {gamma}")

# === Choose Initial Infected Node(s) ===
# Pick one node with high degree or from the top
initial_infecteds = ["1"]
print(f"Initial infected node(s): {initial_infecteds}")

# === Run the SIR Simulation ===
print("Running SIR simulation using EoN...")
t, S, I, R = EoN.fast_SIR(G, tau=tau, gamma=gamma, initial_infecteds=initial_infecteds)

# === Display Final Results ===
print(f"Simulation finished. Final statistics:")
print(f"  Final susceptible: {S[-1]}")
print(f"  Final infected: {I[-1]}")
print(f"  Final recovered: {R[-1]}")
print(f"  Total simulation time: {t[-1]:.2f}")

# === Plot the Results ===
print("Plotting the SIR spread over time...")
plt.figure(figsize=(10, 6))
plt.plot(t, S, label="Susceptible", color='blue')
plt.plot(t, I, label="Infected", color='orange')
plt.plot(t, R, label="Recovered", color='green')
plt.xlabel("Time")
plt.ylabel("Number of Nodes")
plt.title("Information Spread Simulation (SIR Model) - Random Node (Node 1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Done.")


# === april_30/node_classification_enhanced.py ===

# FILE: node_classification_enhanced.py

import json
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------------------------------------
# Load Input Files
# ------------------------------------------------------------------------------

with open("node_content.json") as f:
    node_content = json.load(f)

with open("networkx_nodes.json") as f:
    nodes_data = json.load(f)

with open("networkx_edges.json") as f:
    edges_data = json.load(f)

# ------------------------------------------------------------------------------
# Build NetworkX Graph
# ------------------------------------------------------------------------------

G = nx.DiGraph()
for node in nodes_data:
    G.add_node(str(node["id"]))
for edge in edges_data:
    G.add_edge(str(edge["source"]), str(edge["target"]))

# ------------------------------------------------------------------------------
# Compute SNA Metrics (Using Eigenvector Centrality)
# ------------------------------------------------------------------------------

degree = dict(G.degree())
eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
betweenness = nx.betweenness_centrality(G)

# Create DataFrame
sna_df = pd.DataFrame({
    "id": list(degree.keys()),
    "degree": list(degree.values()),
    "eigenvector": [eigenvector.get(n, 0) for n in degree.keys()],
    "betweenness": [betweenness.get(n, 0) for n in degree.keys()]
}).set_index("id")

# Normalize metrics
scaler = MinMaxScaler()
scaled = scaler.fit_transform(sna_df)
norm_df = pd.DataFrame(scaled, index=sna_df.index, columns=[f"{col}_norm" for col in sna_df.columns])
sna_df = sna_df.join(norm_df)

# ------------------------------------------------------------------------------
# Analyze Risk Per Node (Combine ML + SNA, Filter 0-post Nodes)
# ------------------------------------------------------------------------------

results = []
for node in node_content:
    node_id = str(node.get("id"))
    content = node.get("content", [])
    posts_count = len(content)

    if posts_count == 0:
        continue

    attack_count = sum(1 for article in content if article.get("attack", False))
    percent_attack = round((attack_count / posts_count) * 100, 2)

    if percent_attack >= 75:
        label = "red"
    elif percent_attack <= 25:
        label = "green"
    else:
        label = "yellow"

    sna = sna_df.loc[node_id] if node_id in sna_df.index else pd.Series({
        "degree": 0, "eigenvector": 0, "betweenness": 0,
        "degree_norm": 0, "eigenvector_norm": 0, "betweenness_norm": 0
    })

    risk_score = round(
        (percent_attack / 100) * 0.4 +
        (min(posts_count, 100) / 100) * 0.3 +
        sna["eigenvector_norm"] * 0.3,
        4
    )

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

# Save results
with open("node_risk_scores.json", "w") as f:
    json.dump(results, f, indent=4)

print("✅ Scoring complete. Saved to 'node_risk_scores.json'")

# ------------------------------------------------------------------------------
# Print Top 10 Highest-Risk Nodes
# ------------------------------------------------------------------------------

top_10 = sorted(results, key=lambda x: x["risk_score"], reverse=True)[:10]
print("\n🔥 Top 10 Highest-Risk Nodes (excluding zero-post accounts):")
for i, node in enumerate(top_10, 1):
    print(f"{i}. Node {node['id']} — Risk Score: {node['risk_score']} — Attack %: {node['percent_attack']} — Posts: {node['posts_count']}")

# ------------------------------------------------------------------------------
# Visualize the Network (Filtered, Color by Risk Score)
# ------------------------------------------------------------------------------

# Filter graph and assign risk scores to the subgraph
filtered_node_ids = set(node["id"] for node in results)
subgraph = G.subgraph(filtered_node_ids).copy()

for node in results:
    subgraph.nodes[node["id"]]["risk_score"] = node["risk_score"]

# Layout and color mapping
pos = nx.spring_layout(subgraph, seed=42)
scores = [subgraph.nodes[n].get("risk_score", 0) for n in subgraph.nodes()]
norm = mcolors.Normalize(vmin=min(scores), vmax=max(scores))
cmap = plt.colormaps["Reds"]

plt.figure(figsize=(14, 10))
nx.draw_networkx_edges(subgraph, pos, alpha=0.2)

node_colors = [cmap(norm(subgraph.nodes[n]["risk_score"])) for n in subgraph.nodes()]
nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=150)

# Add node labels
labels = {n: n for n in subgraph.nodes()}
nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=8)

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array(scores)
plt.colorbar(sm, ax=plt.gca(), label="Composite Risk Score")

plt.title("Node Risk Visualization (Filtered + Eigenvector Centrality)", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.show()


# === april_30/eon_example_influential_node.py ===

# FILE: eon_example_influential_node.py

import json
import networkx as nx
import EoN
import matplotlib.pyplot as plt

# === Load Nodes and Edges ===
print("Loading nodes and edges...")

with open('networkx_nodes.json') as f:
    node_data = json.load(f)

with open('networkx_edges.json') as f:
    edge_data = json.load(f)

# === Create a Directed Graph ===
G = nx.DiGraph()
print(f"Creating graph...")

# Add nodes (use string IDs to match edge references)
for node in node_data:
    node_id = str(node['id'])
    G.add_node(node_id, label=node.get('label', ''))
print(f"Total nodes added: {G.number_of_nodes()}")

# Add edges (string IDs to match node IDs)
for edge in edge_data:
    source = str(edge['source'])
    target = str(edge['target'])
    G.add_edge(source, target)
print(f"Total edges added: {G.number_of_edges()}")

# === Define SIR Model Parameters ===
tau = 0.2    # Transmission rate
gamma = 0.05 # Recovery rate
print(f"Model parameters set: tau = {tau}, gamma = {gamma}")

# === Choose Initial Infected Node(s) ===
# Use a known influential node (e.g., high eigenvector and degree)
initial_infecteds = ["191"]
print(f"Initial infected node(s): {initial_infecteds}")

# === Run the SIR Simulation ===
print("Running SIR simulation using EoN...")
t, S, I, R = EoN.fast_SIR(G, tau=tau, gamma=gamma, initial_infecteds=initial_infecteds)

# === Display Final Results ===
print(f"Simulation finished. Final statistics:")
print(f"  Final susceptible: {S[-1]}")
print(f"  Final infected: {I[-1]}")
print(f"  Final recovered: {R[-1]}")
print(f"  Total simulation time: {t[-1]:.2f}")

# === Plot the Results ===
print("Plotting the SIR spread over time...")
plt.figure(figsize=(10, 6))
plt.plot(t, S, label="Susceptible", color='blue')
plt.plot(t, I, label="Infected", color='orange')
plt.plot(t, R, label="Recovered", color='green')
plt.xlabel("Time")
plt.ylabel("Number of Nodes")
plt.title("Information Spread Simulation (SIR Model) - Influential Node (Node 191)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("Done.")


# === april_30/anomaly_detection_directed_graph.py ===

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


# === march_12/demo_link_prediction.py ===

# FILE: demo_link_prediction.py

import json
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.link_prediction import (
    jaccard_coefficient,
    adamic_adar_index,
    preferential_attachment,
    resource_allocation_index
)

# === Load Graph from JSON ===
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

# === Draw Network with Predicted Links ===
def draw_predicted_links(G, predicted_links, title):
    pos = nx.spring_layout(G, seed=42, k=0.8, iterations=100)

    plt.figure(figsize=(14, 10))

    nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color='gray')

    predicted_edges = [(u, v) for u, v, score in predicted_links]
    predicted_nodes = set([node for edge in predicted_edges for node in edge])

    nx.draw_networkx_edges(
        G, pos,
        edgelist=predicted_edges,
        edge_color='red',
        style='dashed',
        width=4
    )

    node_sizes = []
    node_colors = []
    for node in G.nodes():
        if node in predicted_nodes:
            node_sizes.append(1000)
            node_colors.append('orange')
        else:
            node_sizes.append(500)
            node_colors.append('lightblue')

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, edgecolors='black')

    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.show()

# === Get Top N Predictions ===
def get_top_predictions(predictions, top_n=10):
    return sorted(predictions, key=lambda x: x[2], reverse=True)[:top_n]

# === Common Neighbors Prediction + Pretty Print ===
def common_neighbors_prediction(G, top_n=10):
    preds = []
    nodes = list(G.nodes())

    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            if not G.has_edge(u, v):
                cn = len(list(nx.common_neighbors(G, u, v)))
                if cn > 0:
                    preds.append((u, v, cn))

    return sorted(preds, key=lambda x: x[2], reverse=True)[:top_n]

def print_common_neighbors_predictions(G, predictions):
    print("\n=== Common Neighbors Predictions ===\n")
    for idx, (u, v, score) in enumerate(predictions, 1):
        u_neighbors = sorted(list(G.neighbors(u)))
        v_neighbors = sorted(list(G.neighbors(v)))
        common = sorted(list(nx.common_neighbors(G, u, v)))

        print(f"Prediction #{idx}:")
        print(f"Node {u} (neighbors: {u_neighbors}) <--> Node {v} (neighbors: {v_neighbors})")
        print(f"Reason: They share {score} common neighbors -> {common}\n")

def print_jaccard_predictions(G, predictions):
    print("\n=== Jaccard Coefficient Predictions ===\n")
    for idx, (u, v, score) in enumerate(predictions, 1):
        u_neighbors = set(G.neighbors(u))
        v_neighbors = set(G.neighbors(v))
        union_size = len(u_neighbors.union(v_neighbors))
        intersection_size = len(u_neighbors.intersection(v_neighbors))

        print(f"Prediction #{idx}:")
        print(f"Node {u} (neighbors: {sorted(u_neighbors)}) <--> Node {v} (neighbors: {sorted(v_neighbors)})")
        print(f"Reason: {intersection_size} shared neighbors out of {union_size} total neighbors (Jaccard Score: {score:.4f})\n")

def print_adamic_adar_predictions(G, predictions):
    print("\n=== Adamic-Adar Index Predictions ===\n")
    for idx, (u, v, score) in enumerate(predictions, 1):
        common = sorted(list(nx.common_neighbors(G, u, v)))

        print(f"Prediction #{idx}:")
        print(f"Node {u} <--> Node {v}")
        print(f"Reason: They share {len(common)} common neighbors {common}, weighted by inverse log degree (Score: {score:.4f})\n")

def print_preferential_attachment_predictions(G, predictions):
    print("\n=== Preferential Attachment Predictions ===\n")
    for idx, (u, v, score) in enumerate(predictions, 1):
        deg_u = G.degree(u)
        deg_v = G.degree(v)

        print(f"Prediction #{idx}:")
        print(f"Node {u} (degree {deg_u}) <--> Node {v} (degree {deg_v})")
        print(f"Reason: Preferential Attachment score = {score} (degree(u) * degree(v))\n")

def print_resource_allocation_predictions(G, predictions):
    print("\n=== Resource Allocation Index Predictions ===\n")
    for idx, (u, v, score) in enumerate(predictions, 1):
        common = sorted(list(nx.common_neighbors(G, u, v)))

        print(f"Prediction #{idx}:")
        print(f"Node {u} <--> Node {v}")
        print(f"Reason: Shared neighbors {common}, with resources allocated inversely by their degree (Score: {score:.4f})\n")

# === MAIN RUN ===

nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

G = create_graph_from_json(nodes_file_path, edges_file_path)

# --- COMMON NEIGHBORS ---
top_cn = common_neighbors_prediction(G, top_n=10)
print_common_neighbors_predictions(G, top_cn)
draw_predicted_links(G, top_cn, "Common Neighbors Link Prediction")

# --- JACCARD ---
jc_predictions = list(jaccard_coefficient(G))
top_jc = get_top_predictions(jc_predictions, top_n=10)
print_jaccard_predictions(G, top_jc)
draw_predicted_links(G, top_jc, "Jaccard Coefficient Link Prediction")

# --- ADAMIC-ADAR ---
aa_predictions = list(adamic_adar_index(G))
top_aa = get_top_predictions(aa_predictions, top_n=10)
print_adamic_adar_predictions(G, top_aa)
draw_predicted_links(G, top_aa, "Adamic-Adar Index Link Prediction")

# --- PREFERENTIAL ATTACHMENT ---
pa_predictions = list(preferential_attachment(G))
top_pa = get_top_predictions(pa_predictions, top_n=10)
print_preferential_attachment_predictions(G, top_pa)
draw_predicted_links(G, top_pa, "Preferential Attachment Link Prediction")

# --- RESOURCE ALLOCATION INDEX ---
ra_predictions = list(resource_allocation_index(G))
top_ra = get_top_predictions(ra_predictions, top_n=10)
print_resource_allocation_predictions(G, top_ra)
draw_predicted_links(G, top_ra, "Resource Allocation Link Prediction")

# === SUMMARY TABLE ===
def summary_table(predictions_dict):
    print("\n=== SUMMARY OF PREDICTED LINKS ===\n")
    print(f"{'Method':<30} | Predicted Links")
    print("-" * 70)
    for method, predictions in predictions_dict.items():
        pairs = [f"({u}, {v})" for u, v, _ in predictions]
        print(f"{method:<30} | {', '.join(pairs)}")

# Create summary dictionary
predictions_summary = {
    "Common Neighbors": top_cn,
    "Jaccard Coefficient": top_jc,
    "Adamic-Adar Index": top_aa,
    "Preferential Attachment": top_pa,
    "Resource Allocation Index": top_ra
}

# Print the summary table
summary_table(predictions_summary)


# === march_12/demo_hierarchical_community_dendrogram.py ===

# FILE: demo_hierarchical_community_dendrogram.py

import json
import networkx as nx
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import numpy as np

# Step 1: Load data and create the graph (as before)
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

# Paths to your JSON files
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Step 2: Create the graph from your data
G = create_graph_from_json(nodes_file_path, edges_file_path)

# Step 3: Compute the distance matrix
# Create adjacency matrix (binary: 1 if edge exists, 0 otherwise)
adj_matrix = nx.to_numpy_array(G)

# Invert adjacency to create a "distance" matrix
# Where connected nodes are 0 distance, unconnected are 1 distance
distance_matrix = 1 - adj_matrix

# Step 4: Fix the diagonal (self-distance should be 0)
np.fill_diagonal(distance_matrix, 0)

# Confirm the diagonal is zero
assert np.allclose(np.diag(distance_matrix), 0), "Diagonal is not zero!"

# Step 5: Convert to condensed distance format (needed for linkage)
condensed_distance = squareform(distance_matrix)

# Step 6: Compute linkage matrix using agglomerative clustering
Z = linkage(condensed_distance, method='average')  # or 'single', 'complete'

# Step 7: Plot dendrogram
plt.figure(figsize=(12, 8))
dendrogram(Z, labels=[str(n) for n in G.nodes()])
plt.title("Hierarchical Dendrogram (Agglomerative Clustering on Your Dataset)")
plt.xlabel("Nodes")
plt.ylabel("Distance (Edge Disconnection)")
plt.show()


# === march_12/demo_hierarchical_community_5.py ===

# FILE: demo_hierarchical_community_5.py

import json
import networkx as nx
import matplotlib.pyplot as plt

# Load data and create the graph (same as before)
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

# Paths to your JSON files
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Create the graph
G = create_graph_from_json(nodes_file_path, edges_file_path)

# Function to draw communities at each step
def draw_communities_step(G, communities, step):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 7))

    # Flatten communities into node -> community index
    node_color_map = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_color_map[node] = idx

    colors = [node_color_map.get(node, -1) for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.viridis, node_size=500)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos)
    plt.title(f"Step {step}: {len(communities)} Communities")
    plt.axis('off')
    plt.show()

# Manual Girvan-Newman step-by-step hierarchical detection
def girvan_newman_step_by_step(G, max_steps=10):
    # Make a copy of the graph so we don't modify the original
    G_copy = G.copy()
    step = 0

    # Initially, one big component (community)
    components = list(nx.connected_components(G_copy))
    print(f"Step {step}: {len(components)} communities")
    for idx, community in enumerate(components):
        print(f"  Community {idx + 1}: {sorted(community)}")
    draw_communities_step(G_copy, components, step)

    while step < max_steps:
        step += 1

        # Compute betweenness centrality of edges
        edge_betweenness = nx.edge_betweenness_centrality(G_copy)
        
        if not edge_betweenness:
            print("No more edges to remove. Stopping.")
            break

        # Find edge(s) with highest betweenness
        max_bw = max(edge_betweenness.values())
        edges_to_remove = [edge for edge, bw in edge_betweenness.items() if bw == max_bw]

        print(f"\nStep {step}: Removing {len(edges_to_remove)} edge(s) with highest betweenness ({max_bw:.4f})")
        for edge in edges_to_remove:
            print(f"  Removing edge: {edge}")

        # Remove edges with highest betweenness
        G_copy.remove_edges_from(edges_to_remove)

        # Get the new connected components (communities)
        components = list(nx.connected_components(G_copy))
        print(f"After removal: {len(components)} communities")
        for idx, community in enumerate(components):
            print(f"  Community {idx + 1}: {sorted(community)}")

        # Draw the current communities
        draw_communities_step(G_copy, components, step)

        # Optional: Stop if you reach some number of communities
        # if len(components) >= 5:
        #     break

    print("\nFinal communities after step-by-step Girvan-Newman:")
    for idx, community in enumerate(components):
        print(f"  Community {idx + 1}: {sorted(community)}")

    return components

# Run the hierarchical community detection demo
final_components = girvan_newman_step_by_step(G, max_steps=5)


# === march_12/demo_louvain_method.py ===

# FILE: demo_louvain_method.py

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


# === march_12/demo_greedy_optimization_5.py ===

# FILE: demo_greedy_optimization_5.py

import json
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict

# Load data and create the graph
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

# Paths to your JSON files
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Create the graph
G = create_graph_from_json(nodes_file_path, edges_file_path)

# Helper function to visualize communities
def draw_communities(G, communities, title):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 7))
    
    # Flatten communities into a node -> community index mapping
    node_color_map = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_color_map[node] = idx

    colors = [node_color_map[node] for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.viridis, node_size=500)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Helper: Compute modularity of the current partition
def calculate_modularity(G, communities):
    m = G.number_of_edges()
    degrees = dict(G.degree())
    Q = 0
    for community in communities:
        for u in community:
            for v in community:
                A = 1 if G.has_edge(u, v) else 0
                expected = degrees[u] * degrees[v] / (2 * m)
                Q += (A - expected)
    return Q / (2 * m)

# Initialize each node as its own community
communities = [{node} for node in G.nodes()]

# Calculate total number of edges
m = G.number_of_edges()

# Calculate initial modularity
current_modularity = calculate_modularity(G, communities)
print(f"Initial modularity: {current_modularity:.4f}\n")

def compute_delta_Q(G, m, communities):
    # Store ΔQ for each pair of communities
    delta_Qs = {}
    degrees = dict(G.degree())

    # Create a mapping from node to its community index
    node_to_community = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_to_community[node] = idx

    # Prepare community degree sums
    community_degrees = defaultdict(int)
    for idx, community in enumerate(communities):
        for node in community:
            community_degrees[idx] += degrees[node]

    # Compute ΔQ for every pair of communities
    for (i, comm_i), (j, comm_j) in combinations(enumerate(communities), 2):
        e_ij = 0  # number of edges between communities i and j
        for node_i in comm_i:
            for node_j in comm_j:
                if G.has_edge(node_i, node_j):
                    e_ij += 1
        
        # ΔQ calculation
        delta_Q = (e_ij / m) - (community_degrees[i] / (2 * m)) * (community_degrees[j] / (2 * m)) * 2
        delta_Qs[(i, j)] = delta_Q
    
    return delta_Qs

# Start the greedy merge process
for step in range(3):  # Let's do 5 merge steps for demo
    print(f"Step {step + 1}:")

    # Compute ΔQ for all pairs of communities
    delta_Qs = compute_delta_Q(G, m, communities)

    # Sort ΔQ values descending
    sorted_deltas = sorted(delta_Qs.items(), key=lambda x: x[1], reverse=True)

    # Pick the best pair to merge (highest ΔQ)
    if not sorted_deltas:
        print("No more pairs to merge.")
        break
    
    (i, j), best_delta_Q = sorted_deltas[0]

    print(f"  Best ΔQ: {best_delta_Q:.4f} by merging communities {i} and {j}")

    if best_delta_Q <= 0:
        print("  No positive ΔQ remaining. Stopping merge process.")
        break

    # Merge communities i and j
    new_community = communities[i].union(communities[j])

    # Remove old communities and add new merged one
    communities = [c for idx, c in enumerate(communities) if idx not in (i, j)]
    communities.append(new_community)

    # Calculate and print the new modularity
    current_modularity = calculate_modularity(G, communities)
    print(f"  New modularity after merge: {current_modularity:.4f}\n")

# Final communities visualization
print(f"Final communities ({len(communities)}):")
for idx, community in enumerate(communities):
    print(f"  Community {idx + 1}: {sorted(community)}")

# Optional: Visualize final communities
draw_communities(G, communities, "Communities After ΔQ Merging Process")


# === march_12/demo_hierarchical_community_10.py ===

# FILE: demo_hierarchical_community_10.py

import json
import networkx as nx
import matplotlib.pyplot as plt

# Load data and create the graph (same as before)
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

# Paths to your JSON files
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Create the graph
G = create_graph_from_json(nodes_file_path, edges_file_path)

# Function to draw communities at each step
def draw_communities_step(G, communities, step):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 7))

    # Flatten communities into node -> community index
    node_color_map = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_color_map[node] = idx

    colors = [node_color_map.get(node, -1) for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.viridis, node_size=500)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos)
    plt.title(f"Step {step}: {len(communities)} Communities")
    plt.axis('off')
    plt.show()

# Manual Girvan-Newman step-by-step hierarchical detection
def girvan_newman_step_by_step(G, max_steps=10):
    # Make a copy of the graph so we don't modify the original
    G_copy = G.copy()
    step = 0

    # Initially, one big component (community)
    components = list(nx.connected_components(G_copy))
    print(f"Step {step}: {len(components)} communities")
    for idx, community in enumerate(components):
        print(f"  Community {idx + 1}: {sorted(community)}")
    draw_communities_step(G_copy, components, step)

    while step < max_steps:
        step += 1

        # Compute betweenness centrality of edges
        edge_betweenness = nx.edge_betweenness_centrality(G_copy)
        
        if not edge_betweenness:
            print("No more edges to remove. Stopping.")
            break

        # Find edge(s) with highest betweenness
        max_bw = max(edge_betweenness.values())
        edges_to_remove = [edge for edge, bw in edge_betweenness.items() if bw == max_bw]

        print(f"\nStep {step}: Removing {len(edges_to_remove)} edge(s) with highest betweenness ({max_bw:.4f})")
        for edge in edges_to_remove:
            print(f"  Removing edge: {edge}")

        # Remove edges with highest betweenness
        G_copy.remove_edges_from(edges_to_remove)

        # Get the new connected components (communities)
        components = list(nx.connected_components(G_copy))
        print(f"After removal: {len(components)} communities")
        for idx, community in enumerate(components):
            print(f"  Community {idx + 1}: {sorted(community)}")

        # Draw the current communities
        draw_communities_step(G_copy, components, step)

        # Optional: Stop early if no edges remain
        if G_copy.number_of_edges() == 0:
            print("No edges left in the graph. Stopping.")
            break

    print("\nFinal communities after step-by-step Girvan-Newman:")
    for idx, community in enumerate(components):
        print(f"  Community {idx + 1}: {sorted(community)}")

    return components

# Run the hierarchical community detection demo with 10 steps
final_components = girvan_newman_step_by_step(G, max_steps=10)


# === march_12/demo_greedy_optimization_cnm.py ===

# FILE: demo_greedy_optimization_cnm.py

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


# === march_12/demo_hierarchical_edge_betweenness.py ===

# FILE: demo_hierarchical_edge_betweenness.py

import json
import networkx as nx
import matplotlib.pyplot as plt

# Step 1: Load data and create the graph (same as before)
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

# Paths to your JSON files (update as needed)
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Step 2: Create the graph
G = create_graph_from_json(nodes_file_path, edges_file_path)

# Step 3: Compute edge betweenness centrality
edge_betweenness = nx.edge_betweenness_centrality(G, normalized=True)

# Step 4: Print edge betweenness centrality, sorted from highest to lowest
print("\n=== Edge Betweenness Centrality ===")
sorted_edges = sorted(edge_betweenness.items(), key=lambda x: x[1], reverse=True)

for idx, (edge, betweenness) in enumerate(sorted_edges):
    print(f"{idx + 1}. Edge {edge} -> Betweenness: {betweenness:.4f}")

# Optional: Draw the graph with edge thickness representing betweenness
def draw_graph_with_edge_betweenness(G, edge_betweenness):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 7))

    # Normalize widths for visibility
    edge_widths = [5 * edge_betweenness[edge] for edge in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray')
    nx.draw_networkx_labels(G, pos)

    plt.title("Graph with Edge Betweenness Highlighted (Edge Width ∝ Betweenness)")
    plt.axis('off')
    plt.show()

# Step 5: Visualize the graph with edge betweenness
draw_graph_with_edge_betweenness(G, edge_betweenness)


# === march_12/generate_student_networkx_data.py ===

# FILE: generate_student_networkx_data.py

import json
import random
import networkx as nx
import os

# === CONFIGURATION ===
NUM_NODES = 100            # Total number of nodes
NUM_COMMUNITIES = 4        # Number of communities/groups
INTRA_COMMUNITY_P = 0.3    # Probability of edges inside communities (higher = denser)
INTER_COMMUNITY_P = 0.01   # Probability of edges between communities (lower = sparser)
OUTPUT_FOLDER = "student_data"

LABEL_TYPES = ['node_a', 'node_b', 'node_c', 'node_d', 'node_e',
               'node_f', 'node_g', 'node_h', 'node_i', 'node_j']

def generate_community_sizes(num_nodes, num_communities):
    """Distribute nodes roughly equally among communities"""
    base_size = num_nodes // num_communities
    sizes = [base_size] * num_communities
    leftover = num_nodes - sum(sizes)
    for i in range(leftover):
        sizes[i] += 1
    return sizes

def generate_strong_community_graph(num_nodes, num_communities, intra_p, inter_p):
    sizes = generate_community_sizes(num_nodes, num_communities)
    probs = [[intra_p if i == j else inter_p for j in range(num_communities)] for i in range(num_communities)]
    
    # Generate a graph using stochastic block model
    G = nx.stochastic_block_model(sizes, probs, seed=random.randint(1, 10000))
    return G

def assign_labels_to_nodes(G):
    nodes = []
    for node_id in G.nodes():
        label = random.choice(LABEL_TYPES)
        node = {
            "id": str(node_id),
            "label": label
        }
        nodes.append(node)
    return nodes

def convert_edges_to_json(G):
    edges = []
    for u, v in G.edges():
        edges.append({
            "source": str(u),
            "target": str(v)
        })
    return edges

def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def generate_strong_community_dataset(student_id, num_nodes=NUM_NODES, num_communities=NUM_COMMUNITIES,
                                      intra_p=INTRA_COMMUNITY_P, inter_p=INTER_COMMUNITY_P):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Generate graph
    G = generate_strong_community_graph(num_nodes, num_communities, intra_p, inter_p)

    # Assign labels and convert edges
    nodes = assign_labels_to_nodes(G)
    edges = convert_edges_to_json(G)

    # File names
    nodes_filename = f"{OUTPUT_FOLDER}/networkx_nodes_{student_id}.json"
    edges_filename = f"{OUTPUT_FOLDER}/networkx_edges_{student_id}.json"

    save_json(nodes, nodes_filename)
    save_json(edges, edges_filename)

    print(f"✅ Strong community dataset generated for Student {student_id}!")
    print(f"Nodes file: {nodes_filename}")
    print(f"Edges file: {edges_filename}")

# === Example: Generate datasets for 5 students ===
for student_id in range(1, 16):
    generate_strong_community_dataset(student_id)


# === march_12/demo_label_proposition.py ===

# FILE: demo_label_proposition.py

import json
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import random

# Load data and create the graph
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

# Paths to your JSON files
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Create the graph
G = create_graph_from_json(nodes_file_path, edges_file_path)

# Manual label propagation algorithm with step-by-step label updates
def manual_label_propagation(G, max_iterations=10):
    # Initialize each node's label to its own node ID
    labels = {node: node for node in G.nodes()}

    print("Initial labels:")
    for node, label in labels.items():
        print(f"  Node {node}: Label {label}")
    print("\n")

    # Perform label propagation
    for iteration in range(max_iterations):
        print(f"=== Iteration {iteration + 1} ===")
        nodes = list(G.nodes())
        random.shuffle(nodes)  # Randomize node order for each iteration
        
        changes = 0  # Track label changes in this iteration
        
        for node in nodes:
            neighbor_labels = [labels[neighbor] for neighbor in G.neighbors(node)]
            
            if not neighbor_labels:
                continue  # No neighbors, skip
            
            # Find the most common label(s)
            label_count = Counter(neighbor_labels)
            max_count = max(label_count.values())
            most_common_labels = [label for label, count in label_count.items() if count == max_count]
            
            # Randomly choose among equally frequent labels
            new_label = random.choice(most_common_labels)
            
            # If the label has changed, update and print
            if labels[node] != new_label:
                print(f"  Node {node} changed label from {labels[node]} to {new_label}")
                labels[node] = new_label
                changes += 1
        
        print(f"Iteration {iteration + 1} completed. {changes} label changes.\n")
        
        # If no label changed, we have converged
        if changes == 0:
            print("Labels have stabilized. Algorithm converged.\n")
            break

    # Group nodes by labels (communities)
    communities = {}
    for node, label in labels.items():
        communities.setdefault(label, set()).add(node)

    print("Final communities:")
    for idx, community in enumerate(communities.values()):
        print(f"  Community {idx + 1}: {sorted(community)}")
    
    return communities

# Run the manual label propagation
final_communities = manual_label_propagation(G)

# Visualize the final communities
def draw_communities(G, communities, title):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 7))

    # Flatten communities into node -> community index
    node_color_map = {}
    for idx, community in enumerate(communities.values()):
        for node in community:
            node_color_map[node] = idx

    colors = [node_color_map[node] for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.viridis, node_size=500)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos)
    plt.title(title)
    plt.axis('off')
    plt.show()

draw_communities(G, final_communities, "Final Communities After Label Propagation")


# === march_12/demo_hierarchical_community_full.py ===

# FILE: demo_hierarchical_community_full.py

import json
import networkx as nx
import matplotlib.pyplot as plt

# Load data and create the graph (same as before)
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

# Paths to your JSON files
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Create the graph
G = create_graph_from_json(nodes_file_path, edges_file_path)

# Function to draw communities (optional final step)
def draw_communities_final(G, communities, title):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 7))

    # Flatten communities into node -> community index
    node_color_map = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_color_map[node] = idx

    colors = [node_color_map.get(node, -1) for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.viridis, node_size=500)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Girvan-Newman: Break the graph until every node is its own community
def girvan_newman_until_isolated(G):
    # Make a copy so we don't modify the original
    G_copy = G.copy()

    step = 0
    total_nodes = len(G_copy.nodes())

    while G_copy.number_of_edges() > 0:
        step += 1
        edge_betweenness = nx.edge_betweenness_centrality(G_copy)

        if not edge_betweenness:
            print("No more edges to remove.")
            break

        # Remove edges with highest betweenness
        max_bw = max(edge_betweenness.values())
        edges_to_remove = [edge for edge, bw in edge_betweenness.items() if bw == max_bw]

        print(f"Step {step}: Removing {len(edges_to_remove)} edge(s) with betweenness {max_bw:.4f}")
        G_copy.remove_edges_from(edges_to_remove)

    # After all edges are removed, get the final components (each node should be isolated)
    final_components = list(nx.connected_components(G_copy))

    print("\nFinal communities (each node is its own component):")
    for idx, community in enumerate(final_components):
        print(f"  Community {idx + 1}: {sorted(community)}")

    # Optional: Draw final isolated nodes
    draw_communities_final(G_copy, final_components, "Final: All Nodes Isolated")

    return final_components

# Run the full split to isolation
final_components = girvan_newman_until_isolated(G)


# === march_12/demo_greedy_optimization_500.py ===

# FILE: demo_greedy_optimization_500.py

import json
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict

# Load data and create the graph
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

# Paths to your JSON files
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Create the graph
G = create_graph_from_json(nodes_file_path, edges_file_path)

# Helper function to visualize communities
def draw_communities(G, communities, title):
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 7))
    
    # Flatten communities into a node -> community index mapping
    node_color_map = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_color_map[node] = idx

    colors = [node_color_map[node] for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.viridis, node_size=500)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Helper: Compute modularity of the current partition
def calculate_modularity(G, communities):
    m = G.number_of_edges()
    degrees = dict(G.degree())
    Q = 0
    for community in communities:
        for u in community:
            for v in community:
                A = 1 if G.has_edge(u, v) else 0
                expected = degrees[u] * degrees[v] / (2 * m)
                Q += (A - expected)
    return Q / (2 * m)

# Initialize each node as its own community
communities = [{node} for node in G.nodes()]

# Calculate total number of edges
m = G.number_of_edges()

# Calculate initial modularity
current_modularity = calculate_modularity(G, communities)
print(f"Initial modularity: {current_modularity:.4f}\n")

def compute_delta_Q(G, m, communities):
    # Store ΔQ for each pair of communities
    delta_Qs = {}
    degrees = dict(G.degree())

    # Create a mapping from node to its community index
    node_to_community = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_to_community[node] = idx

    # Prepare community degree sums
    community_degrees = defaultdict(int)
    for idx, community in enumerate(communities):
        for node in community:
            community_degrees[idx] += degrees[node]

    # Compute ΔQ for every pair of communities
    for (i, comm_i), (j, comm_j) in combinations(enumerate(communities), 2):
        e_ij = 0  # number of edges between communities i and j
        for node_i in comm_i:
            for node_j in comm_j:
                if G.has_edge(node_i, node_j):
                    e_ij += 1
        
        # ΔQ calculation
        delta_Q = (e_ij / m) - (community_degrees[i] / (2 * m)) * (community_degrees[j] / (2 * m)) * 2
        delta_Qs[(i, j)] = delta_Q
    
    return delta_Qs

# Start the greedy merge process
for step in range(500):  # Let's do 5 merge steps for demo
    print(f"Step {step + 1}:")

    # Compute ΔQ for all pairs of communities
    delta_Qs = compute_delta_Q(G, m, communities)

    # Sort ΔQ values descending
    sorted_deltas = sorted(delta_Qs.items(), key=lambda x: x[1], reverse=True)

    # Pick the best pair to merge (highest ΔQ)
    if not sorted_deltas:
        print("No more pairs to merge.")
        break
    
    (i, j), best_delta_Q = sorted_deltas[0]

    print(f"  Best ΔQ: {best_delta_Q:.4f} by merging communities {i} and {j}")

    if best_delta_Q <= 0:
        print("  No positive ΔQ remaining. Stopping merge process.")
        break

    # Merge communities i and j
    new_community = communities[i].union(communities[j])

    # Remove old communities and add new merged one
    communities = [c for idx, c in enumerate(communities) if idx not in (i, j)]
    communities.append(new_community)

    # Calculate and print the new modularity
    current_modularity = calculate_modularity(G, communities)
    print(f"  New modularity after merge: {current_modularity:.4f}\n")

# Final communities visualization
print(f"Final communities ({len(communities)}):")
for idx, community in enumerate(communities):
    print(f"  Community {idx + 1}: {sorted(community)}")

# Optional: Visualize final communities
draw_communities(G, communities, "Communities After ΔQ Merging Process")


# === march_12/demo_comparison_of_methods.py ===

# FILE: demo_comparison_of_methods.py

import json
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
import time
from networkx.algorithms.community import (
    greedy_modularity_communities,
    label_propagation_communities,
    girvan_newman
)
import pandas as pd

# Load the graph from JSON
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

# Helper: Compute modularity from community sets
def compute_modularity_from_communities(G, communities):
    partition = {}
    for cid, community in enumerate(communities):
        for node in community:
            partition[node] = cid
    return community_louvain.modularity(partition, G)

# Helper: Draw final communities
def draw_communities(G, communities, title):
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

# === Run Louvain ===
start = time.time()
louvain_partition = community_louvain.best_partition(G)
end = time.time()
louvain_communities = {}
for node, cid in louvain_partition.items():
    louvain_communities.setdefault(cid, []).append(node)
louvain_communities = list(map(set, louvain_communities.values()))
louvain_modularity = community_louvain.modularity(louvain_partition, G)
louvain_time = end - start

# === Run Greedy Modularity ===
start = time.time()
greedy_communities = list(greedy_modularity_communities(G))
end = time.time()
greedy_modularity = compute_modularity_from_communities(G, greedy_communities)
greedy_time = end - start

# === Run Label Propagation ===
start = time.time()
label_communities = list(label_propagation_communities(G))
end = time.time()
label_modularity = compute_modularity_from_communities(G, label_communities)
label_time = end - start

# === Run Girvan-Newman for 5 levels, select best modularity ===
start = time.time()
gn_generator = girvan_newman(G)
gn_levels = []
gn_modularities = []

try:
    for level in range(5):
        communities = next(gn_generator)
        communities = list(map(set, communities))
        mod_score = compute_modularity_from_communities(G, communities)
        gn_levels.append(communities)
        gn_modularities.append(mod_score)
except StopIteration:
    print("Girvan-Newman completed before reaching 5 levels.")

end = time.time()

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

# === Results Summary Table ===
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
        'Method': f'Girvan-Newman (Best of 5 levels)',
        'Communities': len(best_gn_communities),
        'Modularity': round(best_gn_modularity, 4),
        'Time (s)': round(girvan_time, 4)
    }
])

# === Print the Results Table ===
print("\n=== Community Detection Comparison ===")
print(results)

# === Draw Final Community Plots ===
draw_communities(G, louvain_communities, "Louvain Communities")
draw_communities(G, greedy_communities, "Greedy Modularity Communities")
draw_communities(G, label_communities, "Label Propagation Communities")
draw_communities(G, best_gn_communities, f"Girvan-Newman Communities (Best of 5 Levels)")


