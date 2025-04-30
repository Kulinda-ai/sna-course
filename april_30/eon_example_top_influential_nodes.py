# FILE: eon_example_top_influential_nodes.py

import json
import networkx as nx
import EoN
import matplotlib.pyplot as plt

# ==============================================================================
# STEP 1: Load Graph Data
# ==============================================================================

print("Loading nodes and edges...")
with open('networkx_nodes.json') as f:
    node_data = json.load(f)

with open('networkx_edges.json') as f:
    edge_data = json.load(f)

# ==============================================================================
# STEP 2: Build Directed Graph
# ==============================================================================

G = nx.DiGraph()
print("Creating graph...")

# Add nodes
for node in node_data:
    G.add_node(str(node['id']), label=node.get('label', ''))

# Add directed edges
for edge in edge_data:
    G.add_edge(str(edge['source']), str(edge['target']))

print(f"Total nodes: {G.number_of_nodes()}, Total edges: {G.number_of_edges()}")

# ==============================================================================
# STEP 3: Identify Influential Nodes Using Eigenvector Centrality
# ==============================================================================

print("Calculating eigenvector centrality...")
try:
    # For directed graphs this may fail, so fallback to undirected version
    centrality = nx.eigenvector_centrality_numpy(G)
except:
    centrality = nx.eigenvector_centrality_numpy(G.to_undirected())

# Sort by influence and get top 5 most influential nodes
top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
initial_infecteds = [node for node, _ in top_nodes]
print(f"Top 5 influential nodes by eigenvector centrality: {initial_infecteds}")

# ==============================================================================
# STEP 4: Define and Run SIR Model from Top Influencers
# ==============================================================================

tau = 0.2    # Infection rate
gamma = 0.05 # Recovery rate
print(f"Running SIR simulation with tau={tau}, gamma={gamma}...")

t, S, I, R = EoN.fast_SIR(G, tau=tau, gamma=gamma, initial_infecteds=initial_infecteds)

# ==============================================================================
# STEP 5: Print Simulation Summary
# ==============================================================================

print("Simulation finished.")
print(f"  Final susceptible: {S[-1]}")
print(f"  Final infected: {I[-1]}")
print(f"  Final recovered: {R[-1]}")
print(f"  Total simulation time: {t[-1]:.2f}")

# ==============================================================================
# STEP 6: Plot Time Series Results
# ==============================================================================

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
