# FILE: demo_greedy_optimization_500.py

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
    Load nodes and edges from JSON into an undirected graph.
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

# Set paths to input files
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Load the graph
G = create_graph_from_json(nodes_file_path, edges_file_path)

# ==============================================================================
# STEP 2: Visual Helper for Drawing Communities
# ==============================================================================

def draw_communities(G, communities, title):
    """
    Draw the graph with nodes colored by their community assignment.
    """
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 7))

    # Assign a color index to each node based on its community
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
# STEP 3: Function to Calculate Modularity of a Community Partition
# ==============================================================================

def calculate_modularity(G, communities):
    """
    Computes the modularity Q of a given community partition.
    Higher Q = better separation between communities.
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
# STEP 4: Greedy Merge Strategy Based on ΔQ
# ==============================================================================

# Initialize each node in its own community
communities = [{node} for node in G.nodes()]
m = G.number_of_edges()

# Print starting modularity
current_modularity = calculate_modularity(G, communities)
print(f"Initial modularity: {current_modularity:.4f}\n")

def compute_delta_Q(G, m, communities):
    """
    Computes the modularity gain (ΔQ) from merging each pair of communities.
    """
    delta_Qs = {}
    degrees = dict(G.degree())

    # Map nodes to their community index
    node_to_community = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_to_community[node] = idx

    # Sum degrees per community
    community_degrees = defaultdict(int)
    for idx, community in enumerate(communities):
        for node in community:
            community_degrees[idx] += degrees[node]

    # For every pair of communities, compute ΔQ
    for (i, comm_i), (j, comm_j) in combinations(enumerate(communities), 2):
        e_ij = 0
        for node_i in comm_i:
            for node_j in comm_j:
                if G.has_edge(node_i, node_j):
                    e_ij += 1

        # Modularity gain formula (ΔQ)
        delta_Q = (e_ij / m) - 2 * (community_degrees[i] / (2 * m)) * (community_degrees[j] / (2 * m))
        delta_Qs[(i, j)] = delta_Q

    return delta_Qs

# ==============================================================================
# STEP 5: Run the Greedy Optimization (up to 500 steps)
# ==============================================================================

for step in range(500):  # Try up to 500 merges
    print(f"Step {step + 1}:")

    delta_Qs = compute_delta_Q(G, m, communities)
    sorted_deltas = sorted(delta_Qs.items(), key=lambda x: x[1], reverse=True)

    if not sorted_deltas:
        print("No more pairs to merge.")
        break

    (i, j), best_delta_Q = sorted_deltas[0]

    print(f"  Best ΔQ: {best_delta_Q:.4f} by merging communities {i} and {j}")

    if best_delta_Q <= 0:
        print("  No positive ΔQ remaining. Stopping merge process.")
        break

    # Merge and update community list
    new_community = communities[i].union(communities[j])
    communities = [c for idx, c in enumerate(communities) if idx not in (i, j)]
    communities.append(new_community)

    current_modularity = calculate_modularity(G, communities)
    print(f"  New modularity after merge: {current_modularity:.4f}\n")

# ==============================================================================
# STEP 6: Final Output and Visualization
# ==============================================================================

print(f"Final communities ({len(communities)}):")
for idx, community in enumerate(communities):
    print(f"  Community {idx + 1}: {sorted(community)}")

draw_communities(G, communities, "Communities After ΔQ Merging Process")
