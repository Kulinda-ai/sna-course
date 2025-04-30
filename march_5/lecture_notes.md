
# 🧠 March 5 Lecture Notes – Core Concepts in Social Network Analysis

Welcome to the foundation of network science!  
This session introduces the building blocks of **social network analysis (SNA)** — key concepts and techniques that help us understand influence, connection, and structure within graphs.

---

## 📌 Centrality Metrics — Measuring Node Importance

Centrality metrics help us **rank nodes** based on their structural roles. Different metrics reveal different kinds of "importance."

### 1. Degree Centrality

**Definition**: The number of direct connections a node has.

- Undirected: just count neighbors.
- Directed: track `in_degree` and `out_degree`.

**Use Cases**:
- Popularity in social networks
- Traffic hubs in transportation

**Formula**:
\[
C_D(v) = rac{	ext{deg}(v)}{N - 1}
\]

---

### 2. Closeness Centrality

**Definition**: How close a node is to all others (based on shortest paths).

- Nodes with high closeness can spread info quickly.
- Defined as the inverse of the average shortest path to all other nodes.

**Use Cases**:
- Rumor spreaders
- Efficient communicators

**Formula**:
\[
C_C(v) = rac{1}{\sum_{u 
eq v} d(u, v)}
\]

---

### 3. Betweenness Centrality

**Definition**: How often a node appears on the shortest path between other nodes.

- Measures "bridge" or "broker" roles in the network.
- High betweenness = control over information flow.

**Use Cases**:
- Gatekeepers
- Vulnerability points

**Formula**:
\[
C_B(v) = \sum_{s 
eq v 
eq t} rac{\sigma_{st}(v)}{\sigma_{st}}
\]
Where:
- \(\sigma_{st}\): # of shortest paths from s to t
- \(\sigma_{st}(v)\): # of those paths that pass through v

---

### 4. Eigenvector Centrality

**Definition**: A node is important if it is connected to other important nodes.

- Recursive: it gives more weight to connections with well-connected nodes.
- Similar to PageRank.

**Use Cases**:
- Influence analysis
- Central players in complex systems

**Formula (conceptually)**:
\[
x_v = rac{1}{\lambda} \sum_{u \in M(v)} x_u
\]
Solving this requires linear algebra (eigenvectors).

---

## 📌 Community Detection — Finding Groups in the Network

Community detection helps uncover **clusters** of nodes that are more densely connected to each other than to the rest of the graph.

### 1. Modularity

**Modularity** measures how well a network is divided into communities.

**Formula**:
\[
Q = rac{1}{2m} \sum_{ij} \left[A_{ij} - rac{k_i k_j}{2m}ight] \delta(c_i, c_j)
\]

Where:
- \(A_{ij}\): adjacency matrix
- \(k_i\): degree of node i
- \(\delta(c_i, c_j)\): 1 if nodes i and j are in the same community

---

### 2. Greedy Modularity Optimization (ΔQ Merge)

- Start with every node in its own community.
- Iteratively merge communities that **increase modularity** the most.
- Stop when no positive gain is possible.

📂 Files:
- `demo_greedy_optimization_5.py` (step-by-step)
- `demo_greedy_optimization_500.py` (faster/full)

---

### 3. Louvain Method

- Fast, hierarchical approach.
- Phase 1: Local movement to increase modularity.
- Phase 2: Collapse communities into super-nodes and repeat.

📂 File:
- `demo_louvain_method.py`

---

### 4. Label Propagation

- Each node adopts the **most common label** among its neighbors.
- Converges when no labels change.

📂 File:
- `demo_label_proposition.py` (step-by-step manual version)

---

### 5. Girvan-Newman Algorithm

- Repeatedly removes edges with the **highest betweenness**.
- As edges are removed, the network splits into communities.
- Visualize step-by-step or using dendrograms.

📂 Files:
- `demo_hierarchical_community_5.py`
- `demo_hierarchical_edge_betweenness.py`
- `demo_hierarchical_community_dendrogram.py`

---

## 📌 Link Prediction — Forecasting Future Connections

These algorithms estimate the **likelihood that two unconnected nodes will connect**.

### 1. Common Neighbors
- Nodes with more shared neighbors are more likely to connect.

### 2. Jaccard Coefficient
- Ratio of shared neighbors to total unique neighbors.

### 3. Adamic-Adar Index
- Emphasizes **rare common neighbors** more heavily.

### 4. Preferential Attachment
- Nodes with higher degree are more likely to gain new links.

### 5. Resource Allocation Index
- Similar to Adamic-Adar, but based on inverse degree allocation.

📂 File:
- `demo_link_prediction.py`

---

## 🧪 Practice: Generate Your Own Networks

Use this to simulate graphs with known communities:

📂 File:
- `generate_student_networkx_data.py`

- Based on the **stochastic block model**
- Outputs labeled `networkx_nodes.json` and `networkx_edges.json`

---

## 🛠️ Technical Setup

Install required libraries:
```bash
pip install networkx matplotlib python-louvain
```

Run any script:
```bash
python example_degree_centrality.py
```

View visuals and modify code as needed!

---

## 🎓 Summary

| Concept | Metric or Tool |
|--------|----------------|
| Centrality | Degree, Closeness, Betweenness, Eigenvector |
| Community | Modularity, Louvain, Label Propagation, Girvan-Newman |
| Prediction | Link Prediction algorithms |
| Visualization | Spring layout, community color maps, dendrograms |

---

These are the **core ideas** in network science — everything else builds on them.

