import json
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist, squareform
import itertools
from collections import defaultdict
import re

class ImageNetworkAnalyzer:
    def __init__(self, analysis_file_path, output_file_path):
        self.analysis_file_path = Path(analysis_file_path)
        self.output_file_path = Path(output_file_path)
        self.analysis_data = None
        self.keyword_weight = 0.8  # High weight for keywords
        self.vibe_weight = 0.2     # Low weight for vibe
        self.semantic_threshold = 0.3  # Threshold for semantic similarity
        
    def load_analysis_data(self):
        """Load the artistic analysis data from JSON file"""
        print("Loading artistic analysis data...")
        with open(self.analysis_file_path, 'r', encoding='utf-8') as f:
            self.analysis_data = json.load(f)
        print(f"Loaded {len(self.analysis_data)} images")
        
    def extract_keywords_and_vibes(self):
        """Extract keywords and vibes from the analysis data"""
        image_data = {}
        
        for item in self.analysis_data:
            filename = item['filename']
            
            # Extract keywords with their confidence scores
            keywords = []
            for kw in item.get('keywords', []):
                keywords.append({
                    'keyword': kw['keyword'].lower(),
                    'confidence': kw['confidence']
                })
            
            # Extract vibe
            vibe = item.get('vibe', '').lower()
            
            image_data[filename] = {
                'keywords': keywords,
                'vibe': vibe,
                'description': item.get('description', '')
            }
            
        return image_data
    
    def calculate_keyword_similarity(self, keywords1, keywords2):
        """Calculate similarity between two sets of keywords"""
        if not keywords1 or not keywords2:
            return 0.0
            
        # Extract keyword strings and their confidence scores
        kw1_dict = {kw['keyword']: kw['confidence'] for kw in keywords1}
        kw2_dict = {kw['keyword']: kw['confidence'] for kw in keywords2}
        
        # Find exact matches
        exact_matches = set(kw1_dict.keys()) & set(kw2_dict.keys())
        exact_similarity = sum(
            min(kw1_dict[kw], kw2_dict[kw]) for kw in exact_matches
        )
        
        # Calculate semantic similarity for non-exact matches
        semantic_similarity = 0.0
        
        # Create TF-IDF vectors for semantic comparison
        all_keywords = list(kw1_dict.keys()) + list(kw2_dict.keys())
        if len(all_keywords) > 1:
            try:
                vectorizer = TfidfVectorizer(
                    analyzer='word',
                    ngram_range=(1, 1),
                    min_df=1,
                    max_df=1.0
                )
                
                # Create documents for TF-IDF
                doc1 = ' '.join(kw1_dict.keys())
                doc2 = ' '.join(kw2_dict.keys())
                
                tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
                semantic_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                
            except Exception as e:
                print(f"Warning: Could not calculate semantic similarity: {e}")
                semantic_similarity = 0.0
        
        # Combine exact and semantic similarities
        total_similarity = exact_similarity + (semantic_similarity * 0.5)
        
        # Normalize by the maximum possible similarity
        max_possible = max(
            sum(kw1_dict.values()),
            sum(kw2_dict.values())
        )
        
        if max_possible > 0:
            return total_similarity / max_possible
        return 0.0
    
    def calculate_vibe_similarity(self, vibe1, vibe2):
        """Calculate similarity between two vibes"""
        if not vibe1 or not vibe2:
            return 0.0
            
        # Simple string similarity for vibes
        if vibe1 == vibe2:
            return 1.0
        
        # Check for partial matches
        if vibe1 in vibe2 or vibe2 in vibe1:
            return 0.7
            
        # Use TF-IDF for more complex similarity
        try:
            vectorizer = TfidfVectorizer(
                analyzer='word',
                ngram_range=(1, 2),
                min_df=1,
                max_df=1.0
            )
            
            tfidf_matrix = vectorizer.fit_transform([vibe1, vibe2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
            
        except Exception as e:
            print(f"Warning: Could not calculate vibe similarity: {e}")
            return 0.0
    
    def calculate_edge_weight(self, image1_data, image2_data):
        """Calculate the weight of an edge between two images"""
        # Calculate keyword similarity
        keyword_sim = self.calculate_keyword_similarity(
            image1_data['keywords'], 
            image2_data['keywords']
        )
        
        # Calculate vibe similarity
        vibe_sim = self.calculate_vibe_similarity(
            image1_data['vibe'], 
            image2_data['vibe']
        )
        
        # Combine similarities with weights
        total_weight = (
            self.keyword_weight * keyword_sim + 
            self.vibe_weight * vibe_sim
        )
        
        return {
            'weight': total_weight,
            'keyword_similarity': keyword_sim,
            'vibe_similarity': vibe_sim
        }
    
    def generate_network_edges(self):
        """Generate all edges in the network"""
        print("Generating network edges...")
        
        # Extract data
        image_data = self.extract_keywords_and_vibes()
        image_names = list(image_data.keys())
        
        edges = []
        total_combinations = len(image_names) * (len(image_names) - 1) // 2
        
        print(f"Calculating {total_combinations} edge combinations...")
        
        # Calculate edges for all pairs
        for i, (img1, img2) in enumerate(itertools.combinations(image_names, 2)):
            if i % 1000 == 0:
                print(f"Processed {i}/{total_combinations} edges...")
                
            edge_data = self.calculate_edge_weight(
                image_data[img1], 
                image_data[img2]
            )
            
            # Only include edges with significant weight
            if edge_data['weight'] > 0.01:  # Threshold to filter weak connections
                edges.append({
                    'source': img1,
                    'target': img2,
                    'weight': edge_data['weight'],
                    'keyword_similarity': edge_data['keyword_similarity'],
                    'vibe_similarity': edge_data['vibe_similarity']
                })
        
        print(f"Generated {len(edges)} edges with weight > 0.01")
        return edges
    
    def save_network_data(self, edges):
        """Save the network data to JSON file"""
        network_data = {
            'metadata': {
                'total_edges': len(edges),
                'keyword_weight': self.keyword_weight,
                'vibe_weight': self.vibe_weight,
                'threshold': 0.01
            },
            'edges': edges
        }
        
        with open(self.output_file_path, 'w', encoding='utf-8') as f:
            json.dump(network_data, f, indent=2)
        
        print(f"Network data saved to {self.output_file_path}")
    
    def run_analysis(self):
        """Run the complete network analysis"""
        print("Starting image network analysis...")
        
        # Load data
        self.load_analysis_data()
        
        # Generate edges
        edges = self.generate_network_edges()
        
        # Save results
        self.save_network_data(edges)
        
        print("Analysis complete!")
        
        # Print some statistics
        if edges:
            weights = [edge['weight'] for edge in edges]
            print(f"Edge weight statistics:")
            print(f"  Min: {min(weights):.4f}")
            print(f"  Max: {max(weights):.4f}")
            print(f"  Mean: {np.mean(weights):.4f}")
            print(f"  Median: {np.median(weights):.4f}")

def main():
    # Define file paths
    analysis_file = Path(__file__).parent.parent.parent / "image_analysis" / "images2_analysis" / "artistic_analysis_images2.json"
    output_file = Path(__file__).parent / "network_edges_images2.json"
    
    # Create analyzer and run analysis
    analyzer = ImageNetworkAnalyzer(analysis_file, output_file)
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 