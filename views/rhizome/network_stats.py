import json
from collections import defaultdict
import numpy as np

def analyze_network(file_path):
    """Analyze the network and display statistics"""
    print("Loading network data...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    edges = data['edges']
    metadata = data['metadata']
    
    print(f"\n=== Network Statistics ===")
    print(f"Total edges: {metadata['total_edges']}")
    print(f"Keyword weight: {metadata['keyword_weight']}")
    print(f"Vibe weight: {metadata['vibe_weight']}")
    print(f"Threshold: {metadata['threshold']}")
    
    # Extract all unique nodes
    nodes = set()
    for edge in edges:
        nodes.add(edge['source'])
        nodes.add(edge['target'])
    
    print(f"Total nodes: {len(nodes)}")
    
    # Calculate edge weight statistics
    weights = [edge['weight'] for edge in edges]
    keyword_sims = [edge['keyword_similarity'] for edge in edges]
    vibe_sims = [edge['vibe_similarity'] for edge in edges]
    
    print(f"\n=== Edge Weight Statistics ===")
    print(f"Weight - Min: {min(weights):.4f}, Max: {max(weights):.4f}")
    print(f"Weight - Mean: {np.mean(weights):.4f}, Median: {np.median(weights):.4f}")
    print(f"Weight - Std: {np.std(weights):.4f}")
    
    print(f"\n=== Similarity Statistics ===")
    print(f"Keyword similarity - Min: {min(keyword_sims):.4f}, Max: {max(keyword_sims):.4f}")
    print(f"Keyword similarity - Mean: {np.mean(keyword_sims):.4f}, Median: {np.median(keyword_sims):.4f}")
    
    print(f"Vibe similarity - Min: {min(vibe_sims):.4f}, Max: {max(vibe_sims):.4f}")
    print(f"Vibe similarity - Mean: {np.mean(vibe_sims):.4f}, Median: {np.median(vibe_sims):.4f}")
    
    # Analyze node connectivity
    node_degrees = defaultdict(int)
    for edge in edges:
        node_degrees[edge['source']] += 1
        node_degrees[edge['target']] += 1
    
    degrees = list(node_degrees.values())
    print(f"\n=== Node Connectivity ===")
    print(f"Average degree: {np.mean(degrees):.2f}")
    print(f"Max degree: {max(degrees)}")
    print(f"Min degree: {min(degrees)}")
    
    # Show some example edges with high weights
    print(f"\n=== Top 10 Strongest Connections ===")
    sorted_edges = sorted(edges, key=lambda x: x['weight'], reverse=True)
    for i, edge in enumerate(sorted_edges[:10]):
        print(f"{i+1}. {edge['source']} <-> {edge['target']} (weight: {edge['weight']:.4f})")
        print(f"   Keyword sim: {edge['keyword_similarity']:.4f}, Vibe sim: {edge['vibe_similarity']:.4f}")
    
    # Show some example edges with high keyword similarity
    print(f"\n=== Top 10 Keyword-Based Connections ===")
    sorted_by_keyword = sorted(edges, key=lambda x: x['keyword_similarity'], reverse=True)
    for i, edge in enumerate(sorted_by_keyword[:10]):
        print(f"{i+1}. {edge['source']} <-> {edge['target']} (keyword sim: {edge['keyword_similarity']:.4f})")
        print(f"   Total weight: {edge['weight']:.4f}, Vibe sim: {edge['vibe_similarity']:.4f}")

if __name__ == "__main__":
    analyze_network("network_edges_images2.json") 