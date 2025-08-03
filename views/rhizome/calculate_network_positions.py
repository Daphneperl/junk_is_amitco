import json
import numpy as np
from scipy.spatial.distance import pdist, squareform
import random

def calculate_network_positions(edges_file_path, output_file_path):
    """
    Calculate 3D positions for images based on their network connections using force-directed layout.
    
    Args:
        edges_file_path: Path to the JSON file containing edge data
        output_file_path: Path to save the calculated positions
    """
    
    # Load the edges data
    with open(edges_file_path, 'r') as f:
        data = json.load(f)
    
    edges = data['edges']
    
    # Extract all unique image nodes
    nodes = set()
    for edge in edges:
        nodes.add(edge['source'])
        nodes.add(edge['target'])
    
    nodes = list(nodes)
    node_to_index = {node: i for i, node in enumerate(nodes)}
    
    print(f"Found {len(nodes)} unique images")
    
    # Create adjacency matrix and weight matrix
    n = len(nodes)
    adjacency_matrix = np.zeros((n, n))
    weight_matrix = np.zeros((n, n))
    
    for edge in edges:
        i = node_to_index[edge['source']]
        j = node_to_index[edge['target']]
        weight = edge['weight']
        
        # Make it undirected
        adjacency_matrix[i][j] = 1
        adjacency_matrix[j][i] = 1
        weight_matrix[i][j] = weight
        weight_matrix[j][i] = weight
    
    # Initialize positions randomly in 3D space
    positions = np.random.rand(n, 3) * 1000 - 500  # Random positions between -500 and 500
    
    # Force-directed layout parameters
    iterations = 100
    attraction_strength = 0.1
    repulsion_strength = 1000
    damping = 0.9
    min_distance = 50
    
    print("Starting force-directed layout simulation...")
    
    # Force-directed layout simulation
    for iteration in range(iterations):
        if iteration % 20 == 0:
            print(f"Iteration {iteration}/{iterations}")
        
        # Calculate forces
        forces = np.zeros((n, 3))
        
        # Attraction forces (connected nodes pull each other)
        for i in range(n):
            for j in range(i + 1, n):
                if adjacency_matrix[i][j] > 0:
                    # Calculate distance
                    diff = positions[i] - positions[j]
                    distance = np.linalg.norm(diff)
                    
                    if distance > 0:
                        # Attraction force based on weight
                        weight = weight_matrix[i][j]
                        force_magnitude = attraction_strength * weight * distance
                        force = (diff / distance) * force_magnitude
                        
                        forces[i] -= force
                        forces[j] += force
        
        # Repulsion forces (all nodes repel each other)
        for i in range(n):
            for j in range(i + 1, n):
                diff = positions[i] - positions[j]
                distance = np.linalg.norm(diff)
                
                if distance > 0 and distance < min_distance:
                    # Repulsion force
                    force_magnitude = repulsion_strength / (distance * distance)
                    force = (diff / distance) * force_magnitude
                    
                    forces[i] += force
                    forces[j] -= force
        
        # Apply forces with damping
        velocities = forces * 0.1
        positions += velocities
        
        # Apply damping
        velocities *= damping
        
        # Keep nodes within bounds
        max_pos = 1000
        positions = np.clip(positions, -max_pos, max_pos)
    
    # Create output data structure
    output_data = {
        "metadata": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "iterations": iterations,
            "attraction_strength": attraction_strength,
            "repulsion_strength": repulsion_strength,
            "damping": damping
        },
        "nodes": []
    }
    
    # Add node positions
    for i, node in enumerate(nodes):
        node_data = {
            "id": node,
            "position": {
                "x": float(positions[i][0]),
                "y": float(positions[i][1]),
                "z": float(positions[i][2])
            },
            "connections": []
        }
        
        # Add connection information
        for j, other_node in enumerate(nodes):
            if adjacency_matrix[i][j] > 0:
                node_data["connections"].append({
                    "target": other_node,
                    "weight": float(weight_matrix[i][j])
                })
        
        output_data["nodes"].append(node_data)
    
    # Save the results
    with open(output_file_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Network positions saved to {output_file_path}")
    print(f"Calculated positions for {len(nodes)} images")
    
    # Print some statistics
    positions_array = np.array([node["position"]["x"] for node in output_data["nodes"]])
    print(f"X range: {positions_array.min():.2f} to {positions_array.max():.2f}")
    
    positions_array = np.array([node["position"]["y"] for node in output_data["nodes"]])
    print(f"Y range: {positions_array.min():.2f} to {positions_array.max():.2f}")
    
    positions_array = np.array([node["position"]["z"] for node in output_data["nodes"]])
    print(f"Z range: {positions_array.min():.2f} to {positions_array.max():.2f}")

if __name__ == "__main__":
    # Calculate positions for the images2 network
    edges_file = "network_edges_images2_filtered_0.4.json"
    output_file = "network_positions_images2.json"
    
    calculate_network_positions(edges_file, output_file) 