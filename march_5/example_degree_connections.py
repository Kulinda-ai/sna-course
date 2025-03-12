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

