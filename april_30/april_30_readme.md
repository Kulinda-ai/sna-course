
# 📅 April 30 — Predictive Modeling and Information Spread with Social Networks

**Lecture Theme:** Epidemic Simulation, Anomaly Detection, and Node Risk Classification using Network + ML techniques.

---

## 📦 Required Input Files

- `networkx_nodes.json` – JSON file with all node IDs and optional labels
- `networkx_edges.json` – JSON file with directed or undirected edges
- `node_content.json` – Node-level content (e.g., post counts, attack labeling)

---

## 🧪 Simulation & Analysis Scripts

| File | Description |
|------|-------------|
| `eon_example_random_node.py` | Runs a basic **SIR model** simulation starting from a **random node**. |
| `eon_visualization_influential_node.py` | Runs SIR and visualizes **spread over time** (midpoint, end) from a **high-centrality node**. |
| `eon_example_low_influence_node.py` | Simulates spread from a **non-influential node**, showing weak or stalled diffusion. |
| `eon_example_top_influential_nodes.py` | Runs SIR from the **top 5 eigenvector centrality nodes**, highlighting how hubs accelerate diffusion. |
| `eon_example_top_influential_nodes_rapid_spread.py` | Same as above but with a **very high tau** (e.g. 0.8) to model aggressive virality. |
| `eon_example_low_influence_node_rapid_spread.py` | Shows what happens when **tau is high** but spread begins from a weak node. |
| `anomaly_detection_directed_graph.py` | Uses **Isolation Forest** to detect structural anomalies in a **directed graph** using SNA features. |
| `anomaly_detection_undirected_graph.py` | Same as above, adapted for **undirected graphs** (e.g. co-authorship). |
| `node_classification.py` | Classifies nodes using **attack post percentage**, assigning red/yellow/green labels. |
| `node_classification_enhanced.py` | Combines **attack content + SNA centrality metrics** into a **composite risk score**, and visualizes risk gradient. |

---

## 📊 Outputs

- `node_risk_scores.json` — Node-level risk scoring (risk score, label, percent attack, SNA metrics)
- Printed summaries of:
  - Top 10 high-risk or red nodes
  - Anomalous nodes and their structural features
- Visuals:
  - SIR time series plots
  - Network visualizations with color-coded risk or anomaly status

---

## 🧠 Learning Objectives

- Understand how **SIR epidemic models** work on real-world networks
- Learn how to evaluate **node importance** using **centrality measures**
- Use **machine learning** to detect **structural anomalies**
- Build **composite node risk scoring systems** by combining network + content signals
- Visualize **information spread** and **risk zones** within a network

---

This session provides a hands-on foundation for blending **network science**, **machine learning**, and **visual analytics** in applied forecasting and threat detection.

