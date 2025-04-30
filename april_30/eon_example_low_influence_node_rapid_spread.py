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
