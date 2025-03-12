# Social Network Analysis - Class Code and Datasets

Welcome to the **Social Network Analysis (SNA)** class code repository. This folder contains all code demos, sample datasets, and take-home assignments used throughout the course.

## Folder Structure

The folder is organized by **date**, making it easy to follow along with class sessions or refer back to specific topics. Each date folder includes the code, datasets, and notes relevant to that class session.

```
/CodeFolder
    /march_5
        ├── greedy_modularity_demo.py
        ├── label_propagation_demo.py
        ├── girvan_newman_demo.py
        ├── louvain_community_demo.py
        ├── networkx_nodes.json
        ├── networkx_edges.json
        └── notes.txt
    /march_12
        ├── link_prediction_demo.py
        ├── link_prediction_with_accuracy.py
        ├── networkx_nodes.json
        ├── networkx_edges.json
        └── notes.txt
```

## Code Demos

Each class includes hands-on examples covering:
- Greedy Modularity Optimization
  - Homegrown ΔQ merge process
  - `greedy_modularity_communities()` from NetworkX (Clauset-Newman-Moore Algorithm)
- Label Propagation Algorithm
- Girvan-Newman Hierarchical Community Detection
- Louvain Modularity Maximization
- Link Prediction Algorithms
  - Common Neighbors
  - Jaccard Coefficient
  - Adamic-Adar Index
  - Preferential Attachment
  - Resource Allocation Index
- Step-by-Step ΔQ Calculation and Hierarchical Merging
- Graph Visualizations for communities and link predictions


## Sample Datasets
- Each folder includes sample graph data in JSON format:
  - `networkx_nodes.json`
  - `networkx_edges.json`
- Use these datasets to test and modify code.


## Take-Home Assignment Datasets
- Located in `/student_datasets/`
- Unique datasets generated for each student.
- Designed to practice community detection and link prediction algorithms.

## How to Use the Code

1. Install the required Python libraries:

`pip install networkx matplotlib python-louvain` or you can just run `pip install -r requirements.txt`

2. Run example scripts (from the terminal or an IDE):

3. Explore and modify the dataset files:
- Open the `.json` files to review the node and edge data formats.
- Test your code on different datasets.

## Key Points for Students
- Review the `README.md` in each folder for specific session instructions.
- Modify and experiment with the scripts to deepen your understanding.
- Use the `/student_datasets/` folder for take-home assignments data.

## Support

For any issues or questions related to the code or datasets, contact your instructor or teaching assistant.