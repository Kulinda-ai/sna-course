# Crypto Network Centrality Analysis

This section demonstrates how to use NetworkX to analyze on-chain transaction graphs with centrality metrics, both unweighted and weighted (by transaction value), and compare how node importance changes depending on whether weights are considered.

---

## 📦 1. Dataset: `networkx.json`

This is your core dataset, formatted for use with NetworkX and Cytoscape. It contains:

- `nodes`: Each node represents a wallet address and includes metadata like:
  - `id`: The wallet address
  - `wallet_type`: (e.g. `"personal"`, `"unknown"`, etc.)
  - Transaction counts and timestamps

- `edges`: Each edge represents a transaction and includes:
  - `source` and `target`: Sender and receiver
  - `value`: Transaction size (used as weight)
  - `dust`: Boolean flag indicating very small or insignificant transfers

---

## 📊 2. Initial NetworkX Analysis (`initial_analysis.py`)

This script loads the data, filters out dust transactions, constructs a directed graph, and calculates standard centrality metrics:

- `degree`: number of connections
- `in_degree` / `out_degree`: directional edge counts
- `degree_centrality`: normalized structural centrality
- `pagerank`: influence based on link structure
- `betweenness`: bridge-like importance
- `closeness`: proximity to others

It saves the enriched graph as:

```
initial_analysis.json
```

---

## 🌐 3. Visualize in Cytoscape via Flask (`initial-analysis.html` + `app.py`)

You can launch an interactive visualization by running the Flask app:

```bash
python app.py
```

Then open [http://localhost:5000/initial-analysis](http://localhost:5000/initial-analysis)

This loads `initial_analysis.json` into Cytoscape.js and renders a force-directed layout of the network. You can enhance it further with filters, highlights, and interactivity.

---

## 🔁 4. Weighted vs. Unweighted Centrality Comparison (`centrality_comparison.py`)

This script compares how node importance changes when weights (i.e., transaction values) are taken into account.

### What it does:
- Loads the original `networkx.json`
- Builds two versions of the graph:
  - **Unweighted**: All edges treated equally
  - **Weighted**: Edge weight = `value`
- Filters out all "dust" transactions in both
- Removes nodes with no transactions
- Computes:
  - `pagerank_unweighted` and `pagerank_weighted`
  - `betweenness_unweighted` and `betweenness_weighted`
- Joins results into a single CSV:

```
centrality_comparison.csv
```

This lets you compare how each wallet ranks differently under volume-based vs. structure-only metrics.

---

## 📁 File Overview

```
.
├── app.py                     # Flask app to serve the HTML + data
├── networkx.json             # Base dataset with nodes & edges
├── initial_analysis.py       # Unweighted analysis + export to JSON
├── initial_analysis.json       # Output of initial analysis
├── templates/
│   └── initial-analysis.html # Cytoscape HTML visualization
├── centrality_comparison.py  # Weighted/unweighted comparison script
├── centrality_comparison.csv # Final CSV comparison output
```

---

## 🚀 How to Run Everything

```bash
# Step 1: Run unweighted analysis
python initial_analysis.py

# Step 2: Run Flask to visualize it
python app.py
# Visit http://localhost:5000/initial-analysis

# Step 3: Run weighted/unweighted centrality comparison
python centrality_comparison.py
```

---

## 📌 Example Use Cases

- Explore how node rankings change when considering **transaction volume**
- Identify high-volume hubs that might not appear central in structure-only graphs
- Demonstrate the real-world impact of weights in financial networks

---

## 📃 Dependencies

- Python 3.8+
- `networkx`
- `flask`

Install dependencies:

```bash
pip install networkx flask
```

---

## 📬 Questions or Improvements?

Feel free to open an issue or PR. This is intended for educational and investigative analysis of financial transaction networks using open tools.

---
