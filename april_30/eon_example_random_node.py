# FILE: eon_example_random_node.py

import json
import networkx as nx
import EoN  # Epidemics on Networks: specialized for simulating SIR, SIS, SEIR, etc.
import matplotlib.pyplot as plt

# ==============================================================================
# STEP 1: Load Network Data
# ==============================================================================

print("Loading nodes and edges...")

with open('networkx_nodes.json') as f:
    node_data = json.load(f)

with open('networkx_edges.json') as f:
    edge_data = json.load(f)

# ==============================================================================
# STEP 2: Create a Directed Graph
# ==============================================================================

G = nx.DiGraph()  # Use DiGraph to allow for directed edges
print("Creating graph...")

# Add nodes
for node in node_data:
    node_id = str(node['id'])
    G.add_node(node_id, label=node.get('label', ''))

print(f"Total nodes added: {G.number_of_nodes()}")

# Add edges
for edge in edge_data:
    source = str(edge['source'])
    target = str(edge['target'])
    G.add_edge(source, target)

print(f"Total edges added: {G.number_of_edges()}")

# ==============================================================================
# STEP 3: Configure SIR Model Parameters
# ==============================================================================

# tau = transmission rate (probability of infecting a neighbor per unit time)
# gamma = recovery rate (probability of recovering per unit time)
tau = 0.2
gamma = 0.05

print(f"Model parameters set: tau = {tau}, gamma = {gamma}")

# ==============================================================================
# STEP 4: Set Initial Infection Seed
# ==============================================================================

# Starting infection at node "1"
initial_infecteds = ["1"]
print(f"Initial infected node(s): {initial_infecteds}")

# ==============================================================================
# STEP 5: Run the Simulation
# ==============================================================================

print("Running SIR simulation using EoN...")

# Run the simulation using the Fast SIR implementation from EoN
t, S, I, R = EoN.fast_SIR(
    G,
    tau=tau,
    gamma=gamma,
    initial_infecteds=initial_infecteds
)

# ==============================================================================
# STEP 6: Output Summary Statistics
# ==============================================================================

print("Simulation finished. Final statistics:")
print(f"  Final susceptible: {S[-1]}")
print(f"  Final infected: {I[-1]}")
print(f"  Final recovered: {R[-1]}")
print(f"  Total simulation time: {t[-1]:.2f}")

# ==============================================================================
# STEP 7: Plot SIR Time Series
# ==============================================================================

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
