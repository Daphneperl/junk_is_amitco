#!/usr/bin/env python3
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
import random
from collections import defaultdict

def load_artistic_analysis():
    """Load the filtered artistic analysis data"""
    with open('image_analysis/images2_analysis/artistic_analysis_images2_filtered.json', 'r') as f:
        return json.load(f)

def create_keyword_vectors(analysis_data):
    """Create keyword vectors for each image"""
    # Collect all unique keywords
    all_keywords = set()
    for item in analysis_data:
        for kw in item['keywords']:
            all_keywords.add(kw['keyword'])
    
    all_keywords = list(all_keywords)
    keyword_to_idx = {kw: i for i, kw in enumerate(all_keywords)}
    
    # Create vectors for each image
    image_vectors = {}
    for item in analysis_data:
        filename = item['filename']
        vector = np.zeros(len(all_keywords))
        
        for kw in item['keywords']:
            keyword = kw['keyword']
            confidence = kw['confidence']
            if keyword in keyword_to_idx:
                vector[keyword_to_idx[keyword]] = confidence
        
        image_vectors[filename] = vector
    
    return image_vectors, all_keywords

def create_vibe_vectors(analysis_data):
    """Create vibe vectors for each image"""
    vibes = ['serene', 'warm', 'neutral', 'mysterious', 'cool', 'raw', 'melancholic', 'whimsical', 'dark']
    vibe_to_idx = {vibe: i for i, vibe in enumerate(vibes)}
    
    image_vibe_vectors = {}
    for item in analysis_data:
        filename = item['filename']
        vibe = item['vibe']
        vector = np.zeros(len(vibes))
        if vibe in vibe_to_idx:
            vector[vibe_to_idx[vibe]] = 1.0
        image_vibe_vectors[filename] = vector
    
    return image_vibe_vectors, vibes

def calculate_similarities(image_vectors, image_vibe_vectors):
    """Calculate similarities between all image pairs"""
    filenames = list(image_vectors.keys())
    n_images = len(filenames)
    
    similarities = {}
    
    for i in range(n_images):
        for j in range(i + 1, n_images):
            img1, img2 = filenames[i], filenames[j]
            
            # Calculate keyword similarity
            vec1 = image_vectors[img1]
            vec2 = image_vectors[img2]
            keyword_sim = cosine_similarity([vec1], [vec2])[0][0]
            
            # Calculate vibe similarity
            vibe1 = image_vibe_vectors[img1]
            vibe2 = image_vibe_vectors[img2]
            vibe_sim = cosine_similarity([vibe1], [vibe2])[0][0]
            
            # Combined similarity (weighted average)
            combined_sim = 0.7 * keyword_sim + 0.3 * vibe_sim
            
            similarities[f"{img1}-{img2}"] = {
                "similarity": float(combined_sim),
                "keyword_similarity": float(keyword_sim),
                "vibe_similarity": float(vibe_sim)
            }
    
    return similarities

def generate_positions(analysis_data, similarities):
    """Generate 3D positions for images using MDS"""
    filenames = [item['filename'] for item in analysis_data]
    n_images = len(filenames)
    
    # Create similarity matrix
    sim_matrix = np.zeros((n_images, n_images))
    filename_to_idx = {filename: i for i, filename in enumerate(filenames)}
    
    for pair, sim_data in similarities.items():
        img1, img2 = pair.split('-')
        if img1 in filename_to_idx and img2 in filename_to_idx:
            i, j = filename_to_idx[img1], filename_to_idx[img2]
            sim_matrix[i][j] = sim_data['similarity']
            sim_matrix[j][i] = sim_data['similarity']
    
    # Set diagonal to 1 (self-similarity)
    np.fill_diagonal(sim_matrix, 1.0)
    
    # Use MDS to create 3D positions
    mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
    # Convert similarity to distance (1 - similarity)
    distances = 1 - sim_matrix
    positions_3d = mds.fit_transform(distances)
    
    # Scale positions to reasonable range
    positions_3d *= 100
    
    # Create positions dictionary
    positions = {}
    for i, filename in enumerate(filenames):
        positions[filename] = {
            "x": float(positions_3d[i][0]),
            "y": float(positions_3d[i][1]),
            "z": float(positions_3d[i][2])
        }
    
    return positions

def generate_connections(similarities, threshold=0.3, max_connections_per_node=5):
    """Generate network connections based on similarities"""
    # Sort similarities by strength
    sorted_pairs = sorted(similarities.items(), key=lambda x: x[1]['similarity'], reverse=True)
    
    connections = []
    connection_counts = defaultdict(int)
    
    for pair, sim_data in sorted_pairs:
        img1, img2 = pair.split('-')
        
        # Check if we should add this connection
        if (sim_data['similarity'] >= threshold and 
            connection_counts[img1] < max_connections_per_node and 
            connection_counts[img2] < max_connections_per_node):
            
            connections.append({
                "source": img1,
                "target": img2,
                "similarity": sim_data['similarity'],
                "keyword_similarity": sim_data['keyword_similarity'],
                "vibe_similarity": sim_data['vibe_similarity']
            })
            
            connection_counts[img1] += 1
            connection_counts[img2] += 1
    
    return connections

def main():
    print("Loading artistic analysis...")
    analysis_data = load_artistic_analysis()
    print(f"Loaded {len(analysis_data)} images")
    
    print("Creating keyword vectors...")
    image_vectors, keywords = create_keyword_vectors(analysis_data)
    print(f"Created vectors for {len(keywords)} unique keywords")
    
    print("Creating vibe vectors...")
    image_vibe_vectors, vibes = create_vibe_vectors(analysis_data)
    print(f"Created vectors for {len(vibes)} vibes")
    
    print("Calculating similarities...")
    similarities = calculate_similarities(image_vectors, image_vibe_vectors)
    print(f"Calculated {len(similarities)} pairwise similarities")
    
    print("Generating positions...")
    positions = generate_positions(analysis_data, similarities)
    
    print("Generating connections...")
    connections = generate_connections(similarities, threshold=0.2, max_connections_per_node=8)
    print(f"Generated {len(connections)} connections")
    
    # Create output data
    output_data = {
        "positions": positions,
        "connections": connections,
        "metadata": {
            "total_images": len(analysis_data),
            "total_connections": len(connections),
            "keywords_count": len(keywords),
            "vibes": vibes
        }
    }
    
    # Save to file
    with open('views/rhizome/similarity_positions_images2.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("Network data saved to views/rhizome/similarity_positions_images2.json")
    
    # Print some statistics
    sim_values = [sim['similarity'] for sim in similarities.values()]
    print(f"Similarity range: {min(sim_values):.3f} - {max(sim_values):.3f}")
    print(f"Average similarity: {np.mean(sim_values):.3f}")
    print(f"Connection threshold: 0.2")
    print(f"Max connections per node: 8")

if __name__ == "__main__":
    main() 