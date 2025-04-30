# FILE: example_degree_connections.py

import networkx as nx
import json
import pandas as pd

# ==============================================================================
# STEP 1: Find 1st, 2nd, and 3rd Degree Connections for Every Node
# ==============================================================================

def find_connections_by_degree(G):
    """
    For each node in the graph, find:
    - 1st-degree connections: directly connected neighbors
    - 2nd-degree: neighbors of neighbors
    - 3rd-degree: neighbors of 2nd-degree nodes (excluding previous)
    """
    connections_by_degree = {}

    for node in G.nodes():
        # Initialize entry for this node
        connections_by_degree[node] = {'1st': [], '2nd': [], '3rd': []}

        # 1st-degree neighbors (direct connections)
        first_degree = set(G.neighbors(node))
        connections_by_degree[node]['1st'] = list(first_degree)

        # 2nd-degree neighbors
        second_degree = set()
        for neighbor in first_degree:
            second_degree.update(G.neighbors(neighbor))

        # Remove original node and any 1st-degree nodes to isolate 2nd-degree
        second_degree = second_degree - first_degree - {node}
        connections_by_degree[node]['2nd'] = list(second_degree)

        # 3rd-degree neighbors
        third_degree = set()
        for neighbor in second_degree:
            third_degree.update(G.neighbors(neighbor))

        # Again, remove overlap with 1st/2nd-degree and self
        third_degree = third_degree - first_degree - second_degree - {node}
        connections_by_degree[node]['3rd'] = list(third_degree)

    return connections_by_degree

# ==============================================================================
# STEP 2: Convert Output to JSON Format for Export or Inspection
# ==============================================================================

def connections_to_json(G):
    connections_by_degree = find_connections_by_degree(G)
    return json.dumps(connections_by_degree, indent=4)

# ==============================================================================
# STEP 3: Load Graph from JSON Files
# ==============================================================================

def create_graph_from_json(nodes_file_path, edges_file_path):
    # Load node and edge lists
    with open(nodes_file_path, 'r') as file:
        nodes_data = json.load(file)

    with open(edges_file_path, 'r') as file:
        edges_data = json.load(file)

    # Create undirected graph (bi-directional relationships assumed)
    G = nx.Graph()

    # Add nodes with optional labels
    for node in nodes_data:
        G.add_node(node['id'], label=node.get('label', ''))

    # Add edges
    for edge in edges_data:
        G.add_edge(edge['source'], edge['target'])

    return G

# ==============================================================================
# STEP 4: Load Data, Build Graph, Compute Connection Layers
# ==============================================================================

nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'
G = create_graph_from_json(nodes_file_path, edges_file_path)

# Compute connection degrees and print
json_data = connections_to_json(G)
print(json_data)

# Save the results to a file for further use
with open('network_connections_by_degree.json', 'w') as f:
    json.dump(json.loads(json_data), f, indent=4)
