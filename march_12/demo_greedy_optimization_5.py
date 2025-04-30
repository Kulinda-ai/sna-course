# FILE: demo_greedy_optimization_5.py

import json
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict

# ==============================================================================
# STEP 1: Load the Graph from JSON
# ==============================================================================

def create_graph_from_json(nodes_file_path, edges_file_path):
    """
    Load nodes and edges from JSON files and return a NetworkX graph.
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

# Set file paths
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Create the graph
G = create_graph_from_json(nodes_file_path, edges_file_path)

# ==============================================================================
# STEP 2: Visual Helper to Draw Communities
# ==============================================================================

def draw_communities(G, communities, title):
    """
    Draws nodes colored by community using spring layout.
    """
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 7))

    # Assign each node a community color index
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

# ==============================================================================
# STEP 3: Function to Compute Modularity (Q)
# ==============================================================================

def calculate_modularity(G, communities):
    """
    Computes modularity score Q for a given community partition.
    """
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

# ==============================================================================
# STEP 4: Greedy Merging Loop Based on ΔQ
# ==============================================================================

# Initialize each node in its own community
communities = [{node} for node in G.nodes()]
m = G.number_of_edges()

# Print initial modularity
current_modularity = calculate_modularity(G, communities)
print(f"Initial modularity: {current_modularity:.4f}\n")

def compute_delta_Q(G, m, communities):
    """
    Compute the change in modularity (ΔQ) for all pairs of communities.
    """
    delta_Qs = {}
    degrees = dict(G.degree())

    # Map each node to its current community index
    node_to_community = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_to_community[node] = idx

    # Precompute community degree sums
    community_degrees = defaultdict(int)
    for idx, community in enumerate(communities):
        for node in community:
            community_degrees[idx] += degrees[node]

    # Compute ΔQ for every pair of communities
    for (i, comm_i), (j, comm_j) in combinations(enumerate(communities), 2):
        e_ij = 0
        for node_i in comm_i:
            for node_j in comm_j:
                if G.has_edge(node_i, node_j):
                    e_ij += 1

        delta_Q = (e_ij / m) - 2 * (community_degrees[i] / (2 * m)) * (community_degrees[j] / (2 * m))
        delta_Qs[(i, j)] = delta_Q

    return delta_Qs

# Run greedy merging for a few steps
for step in range(3):  # You can increase this for deeper merging
    print(f"Step {step + 1}:")

    delta_Qs = compute_delta_Q(G, m, communities)

    # Sort pairs by ΔQ (modularity gain)
    sorted_deltas = sorted(delta_Qs.items(), key=lambda x: x[1], reverse=True)

    if not sorted_deltas:
        print("No more pairs to merge.")
        break

    (i, j), best_delta_Q = sorted_deltas[0]

    print(f"  Best ΔQ: {best_delta_Q:.4f} by merging communities {i} and {j}")

    if best_delta_Q <= 0:
        print("  No positive ΔQ remaining. Stopping merge process.")
        break

    # Merge the two communities
    new_community = communities[i].union(communities[j])

    # Replace old communities with merged one
    communities = [c for idx, c in enumerate(communities) if idx not in (i, j)]
    communities.append(new_community)

    # Recalculate modularity
    current_modularity = calculate_modularity(G, communities)
    print(f"  New modularity after merge: {current_modularity:.4f}\n")

# ==============================================================================
# STEP 5: Final Output and Visualization
# ==============================================================================

print(f"Final communities ({len(communities)}):")
for idx, community in enumerate(communities):
    print(f"  Community {idx + 1}: {sorted(community)}")

# Visualize the final community partition
draw_communities(G, communities, "Communities After ΔQ Merging Process")
