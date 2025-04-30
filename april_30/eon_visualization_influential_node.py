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
