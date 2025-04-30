
# 📘 Social Network Analysis — Class Code and Datasets

Welcome to the **Social Network Analysis (SNA)** class code repository.  
This repo contains all simulation scripts, anomaly detection pipelines, content classifiers, and synthetic datasets used throughout the course.

---

## 🗂️ Folder Structure

All materials are organized by **lecture date**, making it easy to revisit specific topics and demos.

```
/CodeFolder
    /march_5/
        📎 Community Detection + Centrality Analysis
        🧪 Includes: Greedy Optimization, Louvain, Label Propagation, Girvan-Newman, and Link Prediction
    /march_12/
        🔬 Hierarchical Community Detection, Dendrograms, and Practice Data
        🧠 Includes: Modularity ΔQ, Community Merging, Edge Betweenness, Label Propagation Manual
    /april_16/
        💰 Crypto Network Centrality and Flask-Based Visualization
        🌐 Includes: Weighted/unweighted metrics, JSON graph export, Flask web app for Cytoscape.js
    /april_30/
        🚨 Predictive Modeling, Information Spread, and Anomaly Detection
        📊 Includes: EoN simulations, Isolation Forest, node classification, and risk scoring
    /student_datasets/
        👨‍🎓 Personalized test datasets for homework and in-class exercises
```

Each lecture folder includes:
- Python scripts (`*.py`)
- Input datasets (`networkx_nodes.json`, `networkx_edges.json`, `node_content.json`)
- A `README.md` summary and notes for students

---

## 🚀 Topics Covered

### 📌 Community Detection
- Greedy Modularity Optimization (ΔQ)
- Clauset-Newman-Moore Algorithm (`greedy_modularity_communities`)
- Louvain Modularity
- Label Propagation
- Girvan-Newman (manual and with visualization)
- Dendrograms using SciPy
- Clique detection and community visualization

### 📌 Centrality Metrics
- Degree, Closeness, Betweenness, Eigenvector
- Custom Influencer Scoring
- Weighted vs. Unweighted Comparison (April 16)

### 📌 Predictive Modeling and Simulation
- SIR model with EoN
- Spread from random vs. central nodes
- Visualizing infection over time
- Risk scoring from attack content and centrality

### 📌 Anomaly Detection
- Structural anomaly detection using Isolation Forest
- Directed vs. Undirected detection
- Feature z-scores and ML-based outlier classification

### 📌 Link Prediction
- Common Neighbors
- Jaccard Coefficient
- Adamic-Adar Index
- Preferential Attachment
- Resource Allocation Index

---

## 📦 Sample Datasets

Each folder includes JSON files (for example):
- `networkx_nodes.json` – Node list with optional labels
- `networkx_edges.json` – Edge list (directed or undirected)
- `node_content.json` – Node-level post metadata and labels

In April 16, you’ll also see:
- `networkx.json` – Custom format for weighted transaction graphs
- `initial_analysis.json` – Enriched output from NetworkX

---

## 👨‍🏫 Student Assignments

🗂️ `/student_datasets/`  
This contains randomized datasets for take-home exercises in:
- Community detection
- Link prediction
- Hierarchical analysis

---

## 🛠️ Getting Started

1. Install required libraries:
```bash
pip install -r requirements.txt
```

2. Run any script directly:
```bash
python eon_example_random_node.py
```

3. Edit `.json` files to test on new graph structures.

---

## 📌 Tips for Students

- Read the `README.md` inside each session folder for focused guidance
- All `.py` files are annotated with comments for learning
- You’re encouraged to modify and extend the examples

---

## 🙋‍♂️ Help and Support

If you run into bugs or have questions:
- Check the comments in each script
- Ask your instructor or TA
- Explore documentation for: `networkx`, `matplotlib`, `EoN`, `sklearn`, `flask`

---

Happy analyzing! 📊📈
