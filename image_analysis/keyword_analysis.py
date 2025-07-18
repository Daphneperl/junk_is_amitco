#!/usr/bin/env python3
"""
Script to analyze the most common keywords in artistic_analysis.json
"""

import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def load_artistic_analysis(file_path):
    """Load the artistic analysis JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_keywords(data):
    """Extract all keywords from the data"""
    all_keywords = []
    
    for item in data:
        if 'keywords' in item and isinstance(item['keywords'], list):
            for keyword_obj in item['keywords']:
                if isinstance(keyword_obj, dict) and 'keyword' in keyword_obj:
                    all_keywords.append(keyword_obj['keyword'].lower())
    
    return all_keywords

def analyze_keywords(file_path):
    """Analyze keywords and return statistics"""
    print("Loading artistic analysis data...")
    data = load_artistic_analysis(file_path)
    
    print(f"Total images analyzed: {len(data)}")
    
    # Extract all keywords
    all_keywords = extract_keywords(data)
    print(f"Total keywords found: {len(all_keywords)}")
    
    # Count keyword frequencies
    keyword_counts = Counter(all_keywords)
    
    # Get top keywords
    top_keywords = keyword_counts.most_common(50)
    
    return keyword_counts, top_keywords, len(data)

def print_analysis(keyword_counts, top_keywords, total_images):
    """Print the analysis results"""
    print("\n" + "="*60)
    print("KEYWORD ANALYSIS RESULTS")
    print("="*60)
    
    print(f"\nTotal unique keywords: {len(keyword_counts)}")
    print(f"Total images analyzed: {total_images}")
    
    print(f"\nTop 50 Most Common Keywords:")
    print("-" * 40)
    print(f"{'Rank':<4} {'Keyword':<20} {'Count':<8} {'Percentage':<12}")
    print("-" * 40)
    
    for i, (keyword, count) in enumerate(top_keywords, 1):
        percentage = (count / total_images) * 100
        print(f"{i:<4} {keyword:<20} {count:<8} {percentage:.1f}%")
    
    # Show some interesting statistics
    print(f"\nKeywords appearing in more than 10% of images:")
    print("-" * 40)
    threshold = total_images * 0.1
    frequent_keywords = [(k, v) for k, v in keyword_counts.items() if v > threshold]
    frequent_keywords.sort(key=lambda x: x[1], reverse=True)
    
    for keyword, count in frequent_keywords:
        percentage = (count / total_images) * 100
        print(f"{keyword:<20} {count:<8} {percentage:.1f}%")

def create_visualization(top_keywords, output_file="keyword_analysis.png"):
    """Create a bar chart of the top keywords"""
    try:
        # Prepare data for plotting
        keywords, counts = zip(*top_keywords[:20])  # Top 20 for readability
        
        # Create the plot
        plt.figure(figsize=(15, 10))
        bars = plt.bar(range(len(keywords)), counts, color='skyblue', edgecolor='navy')
        
        # Customize the plot
        plt.title('Top 20 Most Common Keywords in Artistic Analysis', fontsize=16, fontweight='bold')
        plt.xlabel('Keywords', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.xticks(range(len(keywords)), keywords, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved as: {output_file}")
        
    except Exception as e:
        print(f"Could not create visualization: {e}")

def main():
    """Main function"""
    file_path = "artistic_analysis.json"
    
    try:
        # Analyze the data
        keyword_counts, top_keywords, total_images = analyze_keywords(file_path)
        
        # Print results
        print_analysis(keyword_counts, top_keywords, total_images)
        
        # Create visualization
        create_visualization(top_keywords)
        
        # Additional insights
        print(f"\n" + "="*60)
        print("ADDITIONAL INSIGHTS")
        print("="*60)
        
        # Keywords that appear only once
        single_occurrence = [k for k, v in keyword_counts.items() if v == 1]
        print(f"\nKeywords appearing only once: {len(single_occurrence)}")
        if len(single_occurrence) <= 10:
            print("Examples:", ", ".join(single_occurrence))
        else:
            print("Examples:", ", ".join(single_occurrence[:10]) + "...")
        
        # Average keywords per image
        total_keywords = sum(keyword_counts.values())
        avg_keywords = total_keywords / total_images
        print(f"\nAverage keywords per image: {avg_keywords:.1f}")
        
    except FileNotFoundError:
        print(f"Error: Could not find file {file_path}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file {file_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 