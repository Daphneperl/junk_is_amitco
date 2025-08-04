#!/usr/bin/env python3
"""
Script to remove top 10 most frequent keywords from artistic_analysis_images2.json
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

def get_top_keywords(data, top_n=10):
    """Get the top N most frequent keywords"""
    all_keywords = extract_keywords(data)
    keyword_counts = Counter(all_keywords)
    return [keyword for keyword, count in keyword_counts.most_common(top_n)]

def filter_keywords(data, keywords_to_remove):
    """Remove specified keywords from the data"""
    filtered_data = []
    
    for item in data:
        filtered_item = item.copy()
        
        if 'keywords' in filtered_item and isinstance(filtered_item['keywords'], list):
            # Keep only keywords that are not in the removal list
            filtered_keywords = []
            for keyword_obj in filtered_item['keywords']:
                if isinstance(keyword_obj, dict) and 'keyword' in keyword_obj:
                    if keyword_obj['keyword'].lower() not in keywords_to_remove:
                        filtered_keywords.append(keyword_obj)
            
            filtered_item['keywords'] = filtered_keywords
        
        filtered_data.append(filtered_item)
    
    return filtered_data

def save_filtered_data(data, output_file):
    """Save the filtered data to a new JSON file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def print_statistics(original_data, filtered_data, removed_keywords):
    """Print statistics about the filtering process"""
    print("="*80)
    print("FILTERING STATISTICS")
    print("="*80)
    
    print(f"\nRemoved keywords: {', '.join(removed_keywords)}")
    
    # Count keywords before and after
    original_keywords = extract_keywords(original_data)
    filtered_keywords = extract_keywords(filtered_data)
    
    print(f"\nOriginal total keywords: {len(original_keywords)}")
    print(f"Filtered total keywords: {len(filtered_keywords)}")
    print(f"Keywords removed: {len(original_keywords) - len(filtered_keywords)}")
    
    # Count unique keywords
    original_unique = len(set(original_keywords))
    filtered_unique = len(set(filtered_keywords))
    
    print(f"\nOriginal unique keywords: {original_unique}")
    print(f"Filtered unique keywords: {filtered_unique}")
    print(f"Unique keywords removed: {original_unique - filtered_unique}")
    
    # Show new top keywords
    filtered_counts = Counter(filtered_keywords)
    new_top_keywords = filtered_counts.most_common(10)
    
    print(f"\nNew top 10 keywords after filtering:")
    print("-" * 50)
    for i, (keyword, count) in enumerate(new_top_keywords, 1):
        percentage = (count / len(filtered_data)) * 100
        print(f"{i:2d}. {keyword:<20} {count:3d} ({percentage:.1f}%)")

def main():
    """Main function"""
    input_file = "artistic_analysis_images2.json"
    output_file = "artistic_analysis_images2_filtered.json"
    
    try:
        print("Loading original artistic analysis data...")
        original_data = load_artistic_analysis(input_file)
        
        print(f"Total images: {len(original_data)}")
        
        # Get top 10 keywords to remove
        print("Identifying top 10 most frequent keywords...")
        top_keywords = get_top_keywords(original_data, 10)
        print(f"Keywords to remove: {', '.join(top_keywords)}")
        
        # Filter the data
        print("Filtering data...")
        filtered_data = filter_keywords(original_data, top_keywords)
        
        # Save filtered data
        print(f"Saving filtered data to {output_file}...")
        save_filtered_data(filtered_data, output_file)
        
        # Print statistics
        print_statistics(original_data, filtered_data, top_keywords)
        
        print(f"\n✅ Successfully created filtered file: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Could not find file {input_file}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file {input_file}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 