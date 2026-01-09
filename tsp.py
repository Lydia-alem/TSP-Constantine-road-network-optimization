

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import osmnx as ox
import itertools
import math
import time
import warnings
warnings.filterwarnings('ignore')

# Phase 1: TSP Solver (updated for Phase 2)
def solve_tsp_optimal(distance_matrix):
    """
    TSP Solver
    """
    n = len(distance_matrix)

    if n <= 10:
        print(f"Using brute-force search for optimal solution (n={n})...")
        return solve_tsp_bruteforce(distance_matrix)
    else:
        print(f"Using nearest neighbor heuristic (n={n} is too large for brute-force)...")
        return solve_tsp_nearest_neighbor(distance_matrix)

def solve_tsp_bruteforce(distance_matrix):
    """Optimal solution using brute-force (for n ≤ 10)"""
    n = len(distance_matrix)
    cities = list(range(1, n))
    best_tour = None
    best_distance = float('inf')

    for perm in itertools.permutations(cities):
        tour = [0] + list(perm) + [0]
        total_dist = sum(distance_matrix[tour[i]][tour[i+1]] for i in range(len(tour)-1))

        if total_dist < best_distance:
            best_distance = total_dist
            best_tour = tour

    return best_tour, best_distance

def solve_tsp_nearest_neighbor(distance_matrix):
    """Heuristic solution using nearest neighbor"""
    n = len(distance_matrix)
    unvisited = set(range(1, n))
    tour = [0]
    current = 0
    total_distance = 0

    while unvisited:
        nearest = min(unvisited, key=lambda city: distance_matrix[current][city])
        total_distance += distance_matrix[current][nearest]
        tour.append(nearest)
        unvisited.remove(nearest)
        current = nearest

    # Return to start
    tour.append(0)
    total_distance += distance_matrix[current][0]

    return tour, total_distance

# Phase 2: OpenStreetMap Integration
def download_constantine_graph():
    """
    Download the road network of Constantine, Algeria from OpenStreetMap
    """
    

    try:
        # Download the graph (use cache to avoid repeated downloads)
        G = ox.graph_from_place(
            "Constantine, Algeria",
            network_type="drive",
            simplify=True
        )

        print(f"✓ Successfully downloaded graph!")
        print(f"  Number of nodes: {len(G.nodes):,}")
        print(f"  Number of edges: {len(G.edges):,}")

        # Project to UTM for accurate distance calculations
        G_projected = ox.project_graph(G)

        return G_projected

    except Exception as e:
        print(f"Error downloading Constantine graph: {e}")
        print("\nCreating a synthetic graph for demonstration...")
        return create_synthetic_graph()

def create_synthetic_graph():
    """Create a synthetic graph if OSM download fails"""
    print("Creating synthetic road network for demonstration...")

    # Create a simple graph (not multi-graph to avoid edge key issues)
    G = nx.Graph()

    # Add 20 nodes in a grid pattern
    for i in range(5):
        for j in range(4):
            node_id = i * 4 + j
            G.add_node(node_id, x=i*1000, y=j*1000, osmid=node_id)

    # Add edges to create a grid
    for i in range(5):
        for j in range(4):
            node_id = i * 4 + j

            # Connect to right neighbor
            if j < 3:
                right_id = i * 4 + (j + 1)
                length = 1000  # 1km
                G.add_edge(node_id, right_id, length=length)

            # Connect to bottom neighbor
            if i < 4:
                bottom_id = (i + 1) * 4 + j
                length = 1000  # 1km
                G.add_edge(node_id, bottom_id, length=length)

    print(f"Created synthetic graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
    return G

def select_routing_nodes(G, num_nodes=6):
    """
    Select routing nodes from the OSM graph
    """
    print(f"\nSelecting {num_nodes} routing nodes from the graph...")

    # Get all nodes
    all_nodes = list(G.nodes())

    if len(all_nodes) < num_nodes:
        print(f"Warning: Only {len(all_nodes)} nodes available")
        num_nodes = len(all_nodes)

    # Select diverse nodes
    selected_nodes = []

    if len(all_nodes) >= num_nodes:
        # Select nodes spaced apart
        step = max(1, len(all_nodes) // num_nodes)
        selected_nodes = all_nodes[::step][:num_nodes]

        # If not enough, add random nodes
        if len(selected_nodes) < num_nodes:
            remaining = [n for n in all_nodes if n not in selected_nodes]
            additional = min(num_nodes - len(selected_nodes), len(remaining))
            if additional > 0:
                selected_nodes.extend(np.random.choice(remaining, additional, replace=False))

    print(f"Selected {len(selected_nodes)} routing nodes")

    # Show node information
    print("\nSelected Node Details:")
    for i, node_id in enumerate(selected_nodes):
        node_data = G.nodes[node_id]
        print(f"  Node {i}: ID {node_id}")
        if 'x' in node_data and 'y' in node_data:
            print(f"    Coordinates: ({node_data['x']:.1f}, {node_data['y']:.1f})")

    return selected_nodes

def calculate_road_distances_safe(G, nodes):
    """
    Calculate shortest-path road distances between nodes (safe version)
    Handles multi-graph issues by converting to simple graph if needed
    """
    n = len(nodes)
    distance_matrix = np.zeros((n, n))

    print(f"\nCalculating shortest-path distances between {n} nodes...")

    # Check if G is a MultiGraph (has parallel edges)
    if isinstance(G, nx.MultiGraph) or isinstance(G, nx.MultiDiGraph):
        print("Converting multi-graph to simple graph for distance calculations...")
        # Create a simple graph with the minimum edge weight
        G_simple = nx.Graph()

        # Add nodes
        for node in G.nodes():
            G_simple.add_node(node, **G.nodes[node])

        # Add edges with minimum weight
        for u, v, data in G.edges(data=True):
            if G_simple.has_edge(u, v):
                # Keep minimum length if multiple edges exist
                if 'length' in data and data['length'] < G_simple[u][v].get('length', float('inf')):
                    G_simple[u][v]['length'] = data['length']
            else:
                if 'length' in data:
                    G_simple.add_edge(u, v, length=data['length'])
                else:
                    # Estimate length if not available
                    if 'x' in G.nodes[u] and 'y' in G.nodes[u] and 'x' in G.nodes[v] and 'y' in G.nodes[v]:
                        x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
                        x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
                        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        G_simple.add_edge(u, v, length=length)

        G = G_simple
        print(f"Converted to simple graph with {len(G.nodes)} nodes and {len(G.edges)} edges")

    # Calculate distances
    for i in range(n):
        for j in range(i+1, n):
            try:
                # Check if nodes are connected
                if nx.has_path(G, nodes[i], nodes[j]):
                    # Calculate shortest path
                    path_length = nx.shortest_path_length(
                        G,
                        nodes[i],
                        nodes[j],
                        weight='length'
                    )

                    # Convert meters to kilometers
                    distance_km = path_length / 1000.0
                    distance_matrix[i][j] = distance_km
                    distance_matrix[j][i] = distance_km
                else:
                    # If no path, use Euclidean distance as fallback
                    if 'x' in G.nodes[nodes[i]] and 'y' in G.nodes[nodes[i]] and \
                       'x' in G.nodes[nodes[j]] and 'y' in G.nodes[nodes[j]]:
                        x1, y1 = G.nodes[nodes[i]]['x'], G.nodes[nodes[i]]['y']
                        x2, y2 = G.nodes[nodes[j]]['x'], G.nodes[nodes[j]]['y']
                        euclidean_dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 1000.0
                        distance_matrix[i][j] = euclidean_dist * 1.5  # Add 50% for road factor
                        distance_matrix[j][i] = euclidean_dist * 1.5
                    else:
                        distance_matrix[i][j] = 10.0  # Default
                        distance_matrix[j][i] = 10.0

            except Exception as e:
                print(f"  Error between nodes {i} and {j}: {e}")
                # Use Euclidean approximation
                if 'x' in G.nodes[nodes[i]] and 'y' in G.nodes[nodes[i]] and \
                   'x' in G.nodes[nodes[j]] and 'y' in G.nodes[nodes[j]]:
                    x1, y1 = G.nodes[nodes[i]]['x'], G.nodes[nodes[i]]['y']
                    x2, y2 = G.nodes[nodes[j]]['x'], G.nodes[nodes[j]]['y']
                    euclidean_dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 1000.0
                    distance_matrix[i][j] = euclidean_dist * 1.5
                    distance_matrix[j][i] = euclidean_dist * 1.5
                else:
                    distance_matrix[i][j] = 10.0
                    distance_matrix[j][i] = 10.0

    print("✓ Distance matrix calculation complete!")
    return distance_matrix

def plot_constantine_map(G):
    """Plot the Constantine road network"""
    print("\nPlotting road network...")

    try:
        fig, ax = plt.subplots(figsize=(12, 10))

        # Check if it's an OSMnx graph
        if hasattr(G, 'graph'):
            # Plot the graph using OSMnx if available
            ox.plot_graph(
                G,
                ax=ax,
                node_size=0,
                edge_linewidth=0.5,
                edge_color='gray',
                show=False,
                close=False
            )
            title = "Road Network of Constantine, Algeria"
        else:
            # Plot synthetic graph
            pos = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes()}
            nx.draw(G, pos, ax=ax, node_size=10, node_color='blue',
                   edge_color='gray', width=0.5, with_labels=False)
            title = "Synthetic Road Network (Constantine simulation)"

        ax.set_title(title, fontsize=16)
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error plotting graph: {e}")
        # Simple fallback plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, "Graph visualization failed\nCheck networkx/osmnx installation",
               ha='center', va='center', fontsize=14)
        plt.show()

def visualize_tsp_tour(G, nodes, tour, distance_matrix):
    """
    Visualize the TSP tour
    """
    print("\n" + "=" * 70)
    print("Visualizing TSP Tour")
    print("=" * 70)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Get coordinates for nodes
    node_coords = {}
    for i, node_id in enumerate(nodes):
        node_data = G.nodes[node_id]
        if 'x' in node_data and 'y' in node_data:
            node_coords[i] = (node_data['x'], node_data['y'])
        else:
            # Generate coordinates if not available
            node_coords[i] = (i * 1000, i * 1000)

    # Plot 1: Network with tour
    ax1 = axes[0]

    try:
        # Try to plot the graph background
        if hasattr(G, 'graph'):
            ox.plot_graph(
                G,
                ax=ax1,
                node_size=0,
                edge_linewidth=0.3,
                edge_color='lightgray',
                show=False,
                close=False
            )
        else:
            # Plot synthetic graph
            pos = {node: (G.nodes[node]['x'], G.nodes[node]['y']) for node in G.nodes()}
            nx.draw(G, pos, ax=ax1, node_size=0, edge_color='lightgray', width=0.3)
    except:
        # Just draw empty background
        ax1.set_facecolor('whitesmoke')

    # Plot nodes with labels
    for i in range(len(nodes)):
        x, y = node_coords[i]
        color = 'green' if i == 0 else 'red'
        ax1.plot(x, y, 'o', markersize=12, color=color, markeredgecolor='black', markeredgewidth=2)
        ax1.text(x, y, f' {i}', fontsize=12, fontweight='bold', color='black')

    # Connect nodes in tour order
    for k in range(len(tour) - 1):
        i = tour[k]
        j = tour[k + 1]
        x1, y1 = node_coords[i]
        x2, y2 = node_coords[j]

        # Draw line
        ax1.plot([x1, x2], [y1, y2], 'b-', linewidth=3, alpha=0.7)

        # Add arrow halfway
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax1.annotate('→',
                    xy=(mid_x, mid_y),
                    xytext=(mid_x - 100, mid_y),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                    fontsize=16, color='blue')

    total_distance = sum(distance_matrix[tour[i]][tour[i+1]] for i in range(len(tour)-1))
    ax1.set_title(f"TSP Tour on Road Network\n{len(nodes)} nodes, Total: {total_distance:.1f} km",
                  fontsize=12)
    ax1.set_xlabel("X Coordinate")
    ax1.set_ylabel("Y Coordinate")

    # Plot 2: Simplified tour diagram
    ax2 = axes[1]

    # Plot nodes in a circle for better visualization
    radius = 5
    angles = np.linspace(0, 2*np.pi, len(nodes), endpoint=False)
    circle_coords = {}
    for i in range(len(nodes)):
        circle_coords[i] = (radius * np.cos(angles[i]), radius * np.sin(angles[i]))

    # Plot nodes in circle
    for i in range(len(nodes)):
        x, y = circle_coords[i]
        color = 'green' if i == 0 else 'red'
        ax2.plot(x, y, 'o', markersize=14, color=color, markeredgecolor='black', markeredgewidth=2)
        ax2.text(x * 1.1, y * 1.1, f'{i}', fontsize=12, fontweight='bold', ha='center')

    # Connect nodes in tour order with distances
    for k in range(len(tour) - 1):
        i = tour[k]
        j = tour[k + 1]
        x1, y1 = circle_coords[i]
        x2, y2 = circle_coords[j]

        # Draw line
        ax2.plot([x1, x2], [y1, y2], 'b-', linewidth=2, alpha=0.7)

        # Add distance label
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        dist = distance_matrix[i][j]
        ax2.text(mid_x, mid_y, f'{dist:.1f} km',
                fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    ax2.set_title(f"TSP Tour Sequence\n{' → '.join(map(str, tour))}", fontsize=12)
    ax2.set_xlim(-radius*1.5, radius*1.5)
    ax2.set_ylim(-radius*1.5, radius*1.5)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.grid(False)

    plt.tight_layout()
    plt.show()

    # Plot 3: Distance matrix heatmap
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    im = ax3.imshow(distance_matrix, cmap='YlOrRd', aspect='auto')

    # Add text annotations
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                ax3.text(j, i, f'{distance_matrix[i, j]:.1f}',
                        ha="center", va="center", color="black", fontsize=8)

    ax3.set_title("Distance Matrix (km)", fontsize=14)
    ax3.set_xlabel("To Node")
    ax3.set_ylabel("From Node")
    ax3.set_xticks(range(len(nodes)))
    ax3.set_yticks(range(len(nodes)))
    ax3.set_xticklabels(range(len(nodes)))
    ax3.set_yticklabels(range(len(nodes)))
    plt.colorbar(im, ax=ax3, label='Distance (km)')
    plt.tight_layout()
    plt.show()

def print_detailed_results(nodes, tour, distance_matrix):
    """
    Print detailed TSP results
    """
    
    print("TSP SOLUTION DETAILS")
   

    total_distance = sum(distance_matrix[tour[i]][tour[i+1]] for i in range(len(tour)-1))

    print(f"\nNumber of locations: {len(nodes)}")
    print(f"Total tour distance: {total_distance:.2f} km")
    print(f"Optimal tour: {' → '.join(map(str, tour))}")

    print("\nTour Segments (with distances):")
    print("-" * 40)

    segment_distances = []
    for k in range(len(tour) - 1):
        i = tour[k]
        j = tour[k + 1]
        dist = distance_matrix[i][j]
        segment_distances.append(dist)

        arrow = "→" if k < len(tour) - 2 else "↺"
        print(f"  Step {k+1:2d}: Node {i} {arrow} Node {j} : {dist:7.2f} km")

    print("-" * 40)
    print(f"  {'Total:':20} {total_distance:7.2f} km")

    # Statistics
    print("\nTour Statistics:")
    print(f"  Average segment distance: {np.mean(segment_distances):.2f} km")
    print(f"  Longest segment: {np.max(segment_distances):.2f} km")
    print(f"  Shortest segment: {np.min(segment_distances):.2f} km")

def main():
    """
    Main function for Phase 2: TSP with OpenStreetMap
    """
   
    
    print("Real-world TSP with Constantine, Algeria Road Network")
    

    # Step 1: Download Constantine road network
    G = download_constantine_graph()

    if G is None:
        print("Failed to create graph. Exiting.")
        return

    # Plot the road network
    plot_constantine_map(G)

    # Step 2: Select routing nodes
    num_nodes = 6  # Small enough for optimal solution
    routing_nodes = select_routing_nodes(G, num_nodes)

    # Step 3: Calculate actual road network distances
    
    print("CALCULATING ROAD NETWORK DISTANCES")
   

    distance_matrix = calculate_road_distances_safe(G, routing_nodes)

    # Display distance matrix
    print("\nDistance Matrix (kilometers):")
    print("     " + " ".join([f"{i:>8}" for i in range(len(routing_nodes))]))
    for i in range(len(routing_nodes)):
        row_str = f"{i:3}: "
        for j in range(len(routing_nodes)):
            if i == j:
                row_str += "       - "
            else:
                row_str += f" {distance_matrix[i][j]:7.2f}"
        print(row_str)

    # Step 4: Solve TSP
    
    print("SOLVING TRAVELING SALESMAN PROBLEM")
    

    start_time = time.time()
    tour, total_distance = solve_tsp_optimal(distance_matrix)
    solve_time = time.time() - start_time

    if tour is None:
        print("Failed to find TSP solution!")
        return

    print(f"TSP solved in {solve_time:.2f} seconds")

    # Step 5: Display results
    print_detailed_results(routing_nodes, tour, distance_matrix)

    # Step 6: Visualize
    visualize_tsp_tour(G, routing_nodes, tour, distance_matrix)

    # Final summary
    
    
    print(f"   {len(G.nodes)} nodes and {len(G.edges)} edges")
    print(f"   {len(routing_nodes)} routing locations")
    print(f"   {len(routing_nodes)}×{len(routing_nodes)} distance matrix")
    print(f"  the optimal TSP tour of {total_distance:.1f} km")
    print(f"  Visualized tour on road network map")
    print("\nThe TSP solution represents an optimal route visiting all selected")
    print("locations using actual road network distances.")

# Run the main function
if __name__ == "__main__":
   
    print()

    main()