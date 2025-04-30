
# 📅 March 5 — Social Network Analysis (SNA) Lecture

**Lecture Theme:** Exploring core graph metrics, neighborhood connections, community structures, and predicting missing links using `networkx`.

---

## 📦 Required Input Files

- `networkx_nodes.json` – List of nodes with unique IDs and optional labels
- `networkx_edges.json` – List of edges connecting node IDs

---

## 🧠 Analysis Scripts Overview

| File | Description |
|------|-------------|
| `example_network_metrics.py` | Calculates graph-level metrics (density, clustering, etc.), computes centralities, and builds a composite **influencer score** for each node. Exports to JSON. |
| `example_degree_centrality.py` | Ranks nodes by **degree centrality** (number of direct connections). |
| `example_closeness_centrality.py` | Computes **closeness centrality** to measure how quickly a node can reach others. |
| `example_betweenness_centrality.py` | Ranks nodes by **betweenness centrality** (how often they act as bridges). |
| `example_eigenvector_centrality.py` | Scores nodes using **eigenvector centrality** based on how influential their neighbors are. |
| `example_influencer_ranking_normalized.py` | Combines centralities into a **weighted influence score** and visualizes node size by influence. |
| `example_degree_connections.py` | Extracts **1st, 2nd, and 3rd degree** connections per node. |
| `example_cliques_detection.py` | Detects **maximal cliques** (fully connected subgroups). |
| `example_community_types.py` | Runs three community detection methods: **Greedy Modularity**, **Label Propagation**, and **Girvan-Newman**. |
| `example_predicted_connections.py` | Predicts likely future links based on **common neighbors** (triadic closure logic). |

---

## 🗂️ Generated Output Files

- `network_info.json` – Overall graph metrics
- `nodes_info.json` – Node centralities and rankings
- `*_centrality.json` – Centrality scores for closeness, betweenness, degree, eigenvector
- `predicted_current_connections.json` – Suggested node pairs that may form connections

---

## 🎯 Learning Objectives

- Compute and interpret **centrality measures**
- Distinguish between **types of influence**
- Identify and visualize **community structure**
- Predict **potential links** using local neighborhood logic
- Organize and export graph analytics using JSON & Pandas

---

This module prepares students for deeper SNA tasks such as intervention design, influence targeting, and network-based prediction.

