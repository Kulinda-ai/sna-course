
# 📅 March 12 — Social Network Analysis (SNA) Lecture

**Lecture Theme:** Hierarchical Community Detection, Link Prediction, and Practice Dataset Generation

---

## 📦 Required Input Files

- `networkx_nodes.json` – List of nodes with unique IDs and optional labels
- `networkx_edges.json` – List of edges connecting node IDs

---

## 🧠 Analysis Scripts Overview

| File | Description |
|------|-------------|
| `demo_greedy_optimization_5.py` | Step-by-step **greedy modularity merging**, showing how communities form over time based on ΔQ (modularity gain). |
| `demo_greedy_optimization_500.py` | Full greedy optimization of community merging for 500 steps to maximize modularity. |
| `demo_greedy_optimization_cnm.py` | Uses **NetworkX’s built-in Clauset-Newman-Moore algorithm** to detect communities and calculate modularity. |
| `demo_hierarchical_community_5.py` | Manual **Girvan-Newman edge removal** demo for 5 steps. Visualizes how communities form as edges with high betweenness are removed. |
| `demo_hierarchical_community_10.py` | Same as above but runs 10 steps for deeper decomposition. |
| `demo_hierarchical_community_full.py` | Runs Girvan-Newman **until all nodes are isolated**, showing full hierarchy decomposition. |
| `demo_hierarchical_community_dendrogram.py` | Uses **SciPy’s agglomerative clustering** and visualizes a dendrogram based on graph structure (0 = connected, 1 = unconnected). |
| `demo_hierarchical_edge_betweenness.py` | Computes and visualizes **edge betweenness centrality**. Key for understanding what edges are "bridges" in the network. |
| `demo_label_proposition.py` | A **manual implementation of Label Propagation** for community detection. Shows how labels evolve across iterations. |
| `demo_link_prediction.py` | Compares **five link prediction algorithms** (common neighbors, Jaccard, Adamic-Adar, preferential attachment, and resource allocation). Visualizes predicted future links. |
| `demo_louvain_method.py` | Runs **Louvain community detection**, exports JSON, and visualizes communities. |
| `generate_student_networkx_data.py` | Generates **practice datasets** with known community structure using the **stochastic block model**. Outputs 15 different node/edge files. |

---

## 🧪 Output Files

- `communities.json` — Community membership assignments
- `predicted links (visuals)` — Overlay visualizations of predicted or evolving edges
- `student_data/` — Folder with node/edge files for practice sets (`networkx_nodes_1.json`, etc.)

---

## 🎯 Learning Objectives

- Learn and compare **community detection algorithms** (greedy, Louvain, GN, label propagation)
- Visualize **modularity gain (ΔQ)** and structural change through edge removal
- Understand **hierarchical structure** using **edge betweenness** and **dendrograms**
- Explore how to **predict future edges** in a social graph using topological features
- Use **synthetic data generation** to build test cases and assign homework

---

This module is ideal for teaching the theory and intuition behind structural analysis, link prediction, and scalable community detection.

