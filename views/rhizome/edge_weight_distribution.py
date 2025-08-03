import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns

def analyze_edge_weight_distribution(file_path):
    """Analyze the distribution of edge weights in the network"""
    print("Loading network data...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    edges = data['edges']
    weights = [edge['weight'] for edge in edges]
    
    print(f"\n=== Edge Weight Distribution Analysis ===")
    print(f"Total edges: {len(weights)}")
    
    # Basic statistics
    print(f"\n=== Basic Statistics ===")
    print(f"Minimum weight: {min(weights):.6f}")
    print(f"Maximum weight: {max(weights):.6f}")
    print(f"Mean weight: {np.mean(weights):.6f}")
    print(f"Median weight: {np.median(weights):.6f}")
    print(f"Standard deviation: {np.std(weights):.6f}")
    print(f"Variance: {np.var(weights):.6f}")
    
    # Percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\n=== Percentiles ===")
    for p in percentiles:
        value = np.percentile(weights, p)
        print(f"{p}th percentile: {value:.6f}")
    
    # Weight ranges analysis
    print(f"\n=== Weight Range Analysis ===")
    ranges = [
        (0.01, 0.05, "Very Low (0.01-0.05)"),
        (0.05, 0.1, "Low (0.05-0.1)"),
        (0.1, 0.2, "Medium-Low (0.1-0.2)"),
        (0.2, 0.3, "Medium (0.2-0.3)"),
        (0.3, 0.5, "Medium-High (0.3-0.5)"),
        (0.5, 1.0, "High (0.5-1.0)")
    ]
    
    for min_w, max_w, label in ranges:
        count = sum(1 for w in weights if min_w <= w < max_w)
        percentage = (count / len(weights)) * 100
        print(f"{label}: {count} edges ({percentage:.2f}%)")
    
    # Create histogram data
    print(f"\n=== Histogram Data ===")
    bins = 50
    hist, bin_edges = np.histogram(weights, bins=bins, range=(min(weights), max(weights)))
    
    print("Bin edges and counts:")
    for i in range(len(bin_edges) - 1):
        bin_start = bin_edges[i]
        bin_end = bin_edges[i + 1]
        count = hist[i]
        percentage = (count / len(weights)) * 100
        print(f"Bin {i+1}: [{bin_start:.6f}, {bin_end:.6f}) - {count} edges ({percentage:.2f}%)")
    
    # Save distribution data to JSON
    distribution_data = {
        "basic_stats": {
            "min": float(min(weights)),
            "max": float(max(weights)),
            "mean": float(np.mean(weights)),
            "median": float(np.median(weights)),
            "std": float(np.std(weights)),
            "variance": float(np.var(weights))
        },
        "percentiles": {str(p): float(np.percentile(weights, p)) for p in percentiles},
        "weight_ranges": {
            label: {
                "count": count,
                "percentage": float((count / len(weights)) * 100)
            }
            for min_w, max_w, label in ranges
            for count in [sum(1 for w in weights if min_w <= w < max_w)]
        },
        "histogram": {
            "bins": bins,
            "bin_edges": [float(edge) for edge in bin_edges],
            "counts": [int(count) for count in hist],
            "percentages": [float((count / len(weights)) * 100) for count in hist]
        }
    }
    
    # Save to JSON file
    output_file = "edge_weight_distribution.json"
    with open(output_file, 'w') as f:
        json.dump(distribution_data, f, indent=2)
    
    print(f"\nDistribution data saved to {output_file}")
    
    return weights, distribution_data

def create_visualizations(weights, distribution_data):
    """Create visualizations of the weight distribution"""
    try:
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Histogram
        ax1.hist(weights, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Edge Weight')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Edge Weight Distribution (Histogram)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Box plot
        ax2.boxplot(weights, vert=True)
        ax2.set_ylabel('Edge Weight')
        ax2.set_title('Edge Weight Distribution (Box Plot)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Cumulative distribution
        sorted_weights = np.sort(weights)
        cumulative = np.arange(1, len(sorted_weights) + 1) / len(sorted_weights)
        ax3.plot(sorted_weights, cumulative, linewidth=2, color='green')
        ax3.set_xlabel('Edge Weight')
        ax3.set_ylabel('Cumulative Probability')
        ax3.set_title('Cumulative Distribution Function')
        ax3.grid(True, alpha=0.3)
        
        # 4. Weight ranges bar chart
        ranges_data = distribution_data["weight_ranges"]
        labels = list(ranges_data.keys())
        percentages = [ranges_data[label]["percentage"] for label in labels]
        
        bars = ax4.bar(range(len(labels)), percentages, color='orange', alpha=0.7)
        ax4.set_xlabel('Weight Ranges')
        ax4.set_ylabel('Percentage of Edges')
        ax4.set_title('Edge Weight Distribution by Ranges')
        ax4.set_xticks(range(len(labels)))
        ax4.set_xticklabels(labels, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        # Add percentage labels on bars
        for bar, percentage in zip(bars, percentages):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{percentage:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('edge_weight_distribution.png', dpi=300, bbox_inches='tight')
        print("Visualizations saved to edge_weight_distribution.png")
        
        # Show the plot
        plt.show()
        
    except ImportError:
        print("Matplotlib or seaborn not available. Skipping visualizations.")
    except Exception as e:
        print(f"Error creating visualizations: {e}")

def main():
    # Analyze the distribution
    weights, distribution_data = analyze_edge_weight_distribution("network_edges_images2.json")
    
    # Create visualizations
    create_visualizations(weights, distribution_data)
    
    print("\n=== Summary ===")
    print("Edge weight distribution analysis complete!")
    print("Files created:")
    print("- edge_weight_distribution.json: Detailed distribution data")
    print("- edge_weight_distribution.png: Visualization plots")

if __name__ == "__main__":
    main() 