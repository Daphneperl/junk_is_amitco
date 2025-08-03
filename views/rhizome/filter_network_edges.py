import json
import os

def filter_network_edges(input_file, output_file, min_weight=0.4):
    """
    Filter network edges to only include those with weights above the specified threshold.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path to the output JSON file
        min_weight (float): Minimum weight threshold (default: 0.4)
    """
    
    print(f"Reading network edges from {input_file}...")
    
    # Read the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Get the original metadata
    original_metadata = data.get('metadata', {})
    original_edges = data.get('edges', [])
    
    print(f"Original total edges: {len(original_edges)}")
    
    # Filter edges with weights above the threshold
    filtered_edges = [edge for edge in original_edges if edge.get('weight', 0) > min_weight]
    
    print(f"Filtered edges with weight > {min_weight}: {len(filtered_edges)}")
    
    # Create new metadata
    new_metadata = original_metadata.copy()
    new_metadata['total_edges'] = len(filtered_edges)
    new_metadata['filtered_threshold'] = min_weight
    new_metadata['original_total_edges'] = original_metadata.get('total_edges', len(original_edges))
    
    # Create the filtered data structure
    filtered_data = {
        'metadata': new_metadata,
        'edges': filtered_edges
    }
    
    # Write the filtered data to the output file
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    print(f"Filtered network edges saved to {output_file}")
    
    # Print some statistics
    if filtered_edges:
        weights = [edge['weight'] for edge in filtered_edges]
        print(f"Filtered edge weight statistics:")
        print(f"  Min weight: {min(weights):.3f}")
        print(f"  Max weight: {max(weights):.3f}")
        print(f"  Average weight: {sum(weights)/len(weights):.3f}")
    
    return len(filtered_edges)

if __name__ == "__main__":
    input_file = "network_edges_images2.json"
    output_file = "network_edges_images2_filtered_0.4.json"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        exit(1)
    
    # Filter the edges
    filtered_count = filter_network_edges(input_file, output_file, min_weight=0.4)
    
    print(f"\nFiltering complete! {filtered_count} edges with weight > 0.4 saved to {output_file}") 