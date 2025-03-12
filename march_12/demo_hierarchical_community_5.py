import json
import networkx as nx
import matplotlib.pyplot as plt

# Load data and create the graph (same as before)
def create_graph_from_json(nodes_file_path, edges_file_path):
    with open(nodes_file_path, 'r') as file:
        nodes_data = json.load(file)
    
    with open(edges_file_path, 'r') as file:
        edges_data = json.load(file)
    
    G = nx.Graph()

    for node in nodes_data:
        G.add_node(node['id'], label=node.get('label', ''))
    
    for edge in edges_data:
        G.add_edge(edge['source'], edge['target'])
    
    return G

# Paths to your JSON files
nodes_file_path = 'networkx_nodes.json'
edges_file_path = 'networkx_edges.json'

# Create the graph
G = create_graph_from_json(nodes_file_path, edges_file_path)

# Function to draw communities at each step
def draw_communities_step(G, communities, step):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 7))

    # Flatten communities into node -> community index
    node_color_map = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_color_map[node] = idx

    colors = [node_color_map.get(node, -1) for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.viridis, node_size=500)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos)
    plt.title(f"Step {step}: {len(communities)} Communities")
    plt.axis('off')
    plt.show()

# Manual Girvan-Newman step-by-step hierarchical detection
def girvan_newman_step_by_step(G, max_steps=10):
    # Make a copy of the graph so we don't modify the original
    G_copy = G.copy()
    step = 0

    # Initially, one big component (community)
    components = list(nx.connected_components(G_copy))
    print(f"Step {step}: {len(components)} communities")
    for idx, community in enumerate(components):
        print(f"  Community {idx + 1}: {sorted(community)}")
    draw_communities_step(G_copy, components, step)

    while step < max_steps:
        step += 1

        # Compute betweenness centrality of edges
        edge_betweenness = nx.edge_betweenness_centrality(G_copy)
        
        if not edge_betweenness:
            print("No more edges to remove. Stopping.")
            break

        # Find edge(s) with highest betweenness
        max_bw = max(edge_betweenness.values())
        edges_to_remove = [edge for edge, bw in edge_betweenness.items() if bw == max_bw]

        print(f"\nStep {step}: Removing {len(edges_to_remove)} edge(s) with highest betweenness ({max_bw:.4f})")
        for edge in edges_to_remove:
            print(f"  Removing edge: {edge}")

        # Remove edges with highest betweenness
        G_copy.remove_edges_from(edges_to_remove)

        # Get the new connected components (communities)
        components = list(nx.connected_components(G_copy))
        print(f"After removal: {len(components)} communities")
        for idx, community in enumerate(components):
            print(f"  Community {idx + 1}: {sorted(community)}")

        # Draw the current communities
        draw_communities_step(G_copy, components, step)

        # Optional: Stop if you reach some number of communities
        # if len(components) >= 5:
        #     break

    print("\nFinal communities after step-by-step Girvan-Newman:")
    for idx, community in enumerate(components):
        print(f"  Community {idx + 1}: {sorted(community)}")

    return components

# Run the hierarchical community detection demo
final_components = girvan_newman_step_by_step(G, max_steps=5)