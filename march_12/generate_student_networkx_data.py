# FILE: generate_student_networkx_data.py
# 🔧 Utility for generating synthetic student practice data with community structure

import json
import random
import networkx as nx
import os

# === CONFIGURATION ===
NUM_NODES = 100            # Total number of nodes
NUM_COMMUNITIES = 4        # Number of communities/groups
INTRA_COMMUNITY_P = 0.3    # Probability of edges inside communities
INTER_COMMUNITY_P = 0.01   # Probability of edges between communities
OUTPUT_FOLDER = "student_data"

LABEL_TYPES = ['node_a', 'node_b', 'node_c', 'node_d', 'node_e',
               'node_f', 'node_g', 'node_h', 'node_i', 'node_j']

def generate_community_sizes(num_nodes, num_communities):
    base_size = num_nodes // num_communities
    sizes = [base_size] * num_communities
    leftover = num_nodes - sum(sizes)
    for i in range(leftover):
        sizes[i] += 1
    return sizes

def generate_strong_community_graph(num_nodes, num_communities, intra_p, inter_p):
    sizes = generate_community_sizes(num_nodes, num_communities)
    probs = [[intra_p if i == j else inter_p for j in range(num_communities)] for i in range(num_communities)]
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

    G = generate_strong_community_graph(num_nodes, num_communities, intra_p, inter_p)
    nodes = assign_labels_to_nodes(G)
    edges = convert_edges_to_json(G)

    nodes_filename = f"{OUTPUT_FOLDER}/networkx_nodes_{student_id}.json"
    edges_filename = f"{OUTPUT_FOLDER}/networkx_edges_{student_id}.json"

    save_json(nodes, nodes_filename)
    save_json(edges, edges_filename)

    print(f"✅ Strong community dataset generated for Student {student_id}!")
    print(f"Nodes file: {nodes_filename}")
    print(f"Edges file: {edges_filename}")

# === Example: Generate datasets for 15 students ===
for student_id in range(1, 16):
    generate_strong_community_dataset(student_id)
