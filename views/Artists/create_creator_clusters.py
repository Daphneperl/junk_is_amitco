import csv
import json
import os
from collections import defaultdict

def find_image_file(base_filename, images_path, images2_path):
    """
    Find the actual image file with any extension in both images and images2 folders.
    """
    possible_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.heic', '.HEIC']
    
    # Check images2 folder first
    for ext in possible_extensions:
        full_path = os.path.join(images2_path, base_filename + ext)
        if os.path.exists(full_path):
            return base_filename + ext
    
    # Check images folder
    for ext in possible_extensions:
        full_path = os.path.join(images_path, base_filename + ext)
        if os.path.exists(full_path):
            return base_filename + ext
    
    return None

def create_creator_clusters():
    """
    Create creator-based clusters from df.json and images/images2 folders.
    Includes all artists from the JSON file.
    """
    
    # Read df.json and group images by creator
    creator_clusters = defaultdict(list)
    images_path = "../../images"
    images2_path = "../../images2"
    
    with open('../../assets/df.json', 'r', encoding='utf-8') as jsonfile:
        data = json.load(jsonfile)
        
        for item in data:
            creator = item.get('@creator')
            filename = item.get('filename')
            
            # Skip entries where creator is None or empty
            if not creator:
                continue
                
            # Find the actual image file with extension
            actual_filename = find_image_file(filename, images_path, images2_path)
            
            if actual_filename:
                creator_clusters[creator].append({
                    'filename': actual_filename,
                    'title': item.get('title', ''),
                    'location': item.get('location', ''),
                    'exact_spot': item.get('exact_spot', ''),
                    'intimacy_level': str(item.get('intimacy_level', '')),
                    'date_stamp': item.get('date_stamp', ''),
                    'daytime_icon': item.get('daytime_icon', ''),
                    'rawness_percent': str(item.get('rawness_percent', '')),
                    'context': item.get('context', ''),
                    'artistic_description': item.get('artistic_description', ''),
                    'keywords': item.get('keywords', []),
                    'keyword_confidences': item.get('keyword_confidences', []),
                    'vibe': item.get('vibe', '')
                })
    
    # Convert to list format for easier processing in JavaScript
    clusters_list = []
    for creator, images in creator_clusters.items():
        if len(images) > 0:  # Only include creators with images
            clusters_list.append({
                'creator': creator,
                'images': images
            })
    
    # Sort clusters by number of images (descending)
    clusters_list.sort(key=lambda x: len(x['images']), reverse=True)
    
    # Save to JSON file
    with open('creator_clusters.json', 'w', encoding='utf-8') as jsonfile:
        json.dump(clusters_list, jsonfile, indent=2, ensure_ascii=False)
    
    print(f"Created {len(clusters_list)} creator clusters:")
    for cluster in clusters_list:
        print(f"  {cluster['creator']}: {len(cluster['images'])} images")
    
    return clusters_list

if __name__ == "__main__":
    create_creator_clusters() 