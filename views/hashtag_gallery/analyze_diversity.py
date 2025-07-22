import json
from collections import Counter

def analyze_diversity(filename):
    """Analyze the diversity of image usage in the connections file"""
    with open(filename, 'r', encoding='utf-8') as f:
        connections = json.load(f)
    
    # Count image usage
    image_usage = Counter()
    for data in connections.values():
        image_usage.update(data['matching_images'])
    
    # Calculate statistics
    total_quotes = len(connections)
    total_image_slots = total_quotes * 10
    unique_images = len(image_usage)
    diversity_ratio = unique_images / total_image_slots
    
    print(f"=== DIVERSITY ANALYSIS ===")
    print(f"Total quotes: {total_quotes}")
    print(f"Total image slots: {total_image_slots}")
    print(f"Unique images used: {unique_images}")
    print(f"Diversity ratio: {diversity_ratio:.2%}")
    print()
    
    # Most used images
    print("MOST USED IMAGES:")
    for img, count in image_usage.most_common(10):
        print(f"  {img}: {count} times")
    print()
    
    # Least used images
    print("LEAST USED IMAGES:")
    for img, count in sorted(image_usage.items(), key=lambda x: x[1])[:10]:
        print(f"  {img}: {count} times")
    print()
    
    # Distribution analysis
    usage_counts = Counter(image_usage.values())
    print("USAGE DISTRIBUTION:")
    for usage_count, num_images in sorted(usage_counts.items()):
        print(f"  {usage_count} usage(s): {num_images} images")
    print()
    
    # Sample some quotes to show variety
    print("SAMPLE QUOTES AND THEIR IMAGES:")
    sample_quotes = list(connections.keys())[:5]
    for quote in sample_quotes:
        data = connections[quote]
        print(f"\n{quote}:")
        print(f"  Theme: {data['bottom_line']}")
        print(f"  Images: {', '.join(data['matching_images'][:5])}...")
        if len(data['matching_images']) > 5:
            print(f"  ... and {len(data['matching_images']) - 5} more")

if __name__ == "__main__":
    analyze_diversity("quote_to_images_connections.json") 