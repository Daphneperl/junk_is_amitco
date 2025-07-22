import json
import random

def load_data():
    """Load the connections and artistic analysis data"""
    with open('quote_to_images_connections.json', 'r', encoding='utf-8') as f:
        connections = json.load(f)
    
    with open('../../image_analysis/artistic_analysis_filtered.json', 'r', encoding='utf-8') as f:
        artistic_analysis = json.load(f)
    
    return connections, artistic_analysis

def find_image_data(filename, artistic_analysis):
    """Find image data by filename"""
    for img in artistic_analysis:
        if img['filename'] == filename:
            return img
    return None

def print_detailed_connections(connections, artistic_analysis, num_examples=5):
    """Print detailed examples of connections"""
    print("=" * 100)
    print("DETAILED SEMANTIC CONNECTIONS BETWEEN QUOTES AND IMAGES")
    print("=" * 100)
    
    # Get random sample of quotes
    sample_quotes = random.sample(list(connections.keys()), num_examples)
    
    for i, quote in enumerate(sample_quotes, 1):
        data = connections[quote]
        print(f"\n{i}. QUOTE: {quote}")
        print(f"   Bottom Line: {data['bottom_line']}")
        print(f"   Origin: {data['origin']}")
        print(f"   Connected Images ({data['num_images']}):")
        
        # Show first 3 images with details
        for j, img_filename in enumerate(data['matching_images'][:3], 1):
            img_data = find_image_data(img_filename, artistic_analysis)
            if img_data:
                print(f"     {j}. {img_filename}")
                print(f"        Description: {img_data['description']}")
                print(f"        Keywords: {', '.join([kw['keyword'] for kw in img_data['keywords'][:5]])}")
                print(f"        Vibe: {img_data.get('vibe', 'N/A')}")
        
        if len(data['matching_images']) > 3:
            print(f"     ... and {len(data['matching_images']) - 3} more images")
        
        print("-" * 80)

def print_statistics(connections):
    """Print statistics about the connections"""
    print("\n" + "=" * 100)
    print("CONNECTION STATISTICS")
    print("=" * 100)
    
    total_quotes = len(connections)
    total_images = sum(len(data['matching_images']) for data in connections.values())
    avg_images_per_quote = total_images / total_quotes
    
    print(f"Total quotes processed: {total_quotes}")
    print(f"Total images connected: {total_images}")
    print(f"Average images per quote: {avg_images_per_quote:.1f}")
    
    # Count images by bottom line theme
    bottom_line_counts = {}
    for data in connections.values():
        bottom_line = data['bottom_line']
        bottom_line_counts[bottom_line] = bottom_line_counts.get(bottom_line, 0) + 1
    
    print(f"\nBottom Line themes:")
    for theme, count in sorted(bottom_line_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {theme}: {count} quotes")

def main():
    """Main function"""
    print("Loading data...")
    connections, artistic_analysis = load_data()
    
    print(f"Loaded {len(connections)} quote connections and {len(artistic_analysis)} images")
    
    # Print statistics
    print_statistics(connections)
    
    # Print detailed examples
    print_detailed_connections(connections, artistic_analysis, num_examples=8)

if __name__ == "__main__":
    main() 