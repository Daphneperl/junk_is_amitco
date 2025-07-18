#!/usr/bin/env python3
"""
Simple script to analyze the most common keywords in artistic_analysis.json
"""

import json
from collections import Counter

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
    print("\n" + "="*80)
    print("KEYWORD ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\nTotal unique keywords: {len(keyword_counts)}")
    print(f"Total images analyzed: {total_images}")
    
    print(f"\nTop 50 Most Common Keywords:")
    print("-" * 60)
    print(f"{'Rank':<4} {'Keyword':<25} {'Count':<8} {'Percentage':<12}")
    print("-" * 60)
    
    for i, (keyword, count) in enumerate(top_keywords, 1):
        percentage = (count / total_images) * 100
        print(f"{i:<4} {keyword:<25} {count:<8} {percentage:.1f}%")
    
    # Show keywords appearing in more than 10% of images
    print(f"\nKeywords appearing in more than 10% of images:")
    print("-" * 60)
    threshold = total_images * 0.1
    frequent_keywords = [(k, v) for k, v in keyword_counts.items() if v > threshold]
    frequent_keywords.sort(key=lambda x: x[1], reverse=True)
    
    if frequent_keywords:
        for keyword, count in frequent_keywords:
            percentage = (count / total_images) * 100
            print(f"{keyword:<25} {count:<8} {percentage:.1f}%")
    else:
        print("No keywords appear in more than 10% of images")

def print_additional_insights(keyword_counts, total_images):
    """Print additional insights about the data"""
    print(f"\n" + "="*80)
    print("ADDITIONAL INSIGHTS")
    print("="*80)
    
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
    
    # Keywords appearing in more than 5% of images
    threshold_5 = total_images * 0.05
    frequent_5 = [(k, v) for k, v in keyword_counts.items() if v > threshold_5]
    frequent_5.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nKeywords appearing in more than 5% of images ({len(frequent_5)} total):")
    print("-" * 60)
    for keyword, count in frequent_5[:20]:  # Show top 20
        percentage = (count / total_images) * 100
        print(f"{keyword:<25} {count:<8} {percentage:.1f}%")
    
    if len(frequent_5) > 20:
        print(f"... and {len(frequent_5) - 20} more")

def main():
    """Main function"""
    file_path = "artistic_analysis.json"
    
    try:
        # Analyze the data
        keyword_counts, top_keywords, total_images = analyze_keywords(file_path)
        
        # Print results
        print_analysis(keyword_counts, top_keywords, total_images)
        
        # Print additional insights
        print_additional_insights(keyword_counts, total_images)
        
    except FileNotFoundError:
        print(f"Error: Could not find file {file_path}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file {file_path}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 