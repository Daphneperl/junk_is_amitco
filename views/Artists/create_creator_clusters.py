import csv
import json
import os
from collections import defaultdict

def find_image_file(base_filename, images2_path):
    """
    Find the actual image file with any extension in the images2 folder.
    """
    possible_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.heic', '.HEIC']
    
    for ext in possible_extensions:
        full_path = os.path.join(images2_path, base_filename + ext)
        if os.path.exists(full_path):
            return base_filename + ext
    
    return None

def create_creator_clusters():
    """
    Create creator-based clusters from DF.csv and images2 folder.
    Filters out entries where creator is "-" and groups images by creator.
    """
    
    # Read DF.csv and group images by creator
    creator_clusters = defaultdict(list)
    images2_path = "../../images2"
    
    with open('../../assets/DF.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            creator = row['@creator']
            filename = row['filename']
            
            # Skip entries where creator is "-" or empty
            if creator == '-' or not creator:
                continue
                
            # Find the actual image file with extension
            actual_filename = find_image_file(filename, images2_path)
            
            if actual_filename:
                creator_clusters[creator].append({
                    'filename': actual_filename,
                    'title': row['title'],
                    'location': row['location'],
                    'exact_spot': row['exact_spot'],
                    'intimacy_level': row['intimacy_level'],
                    'date_stamp': row['date_stamp'],
                    'daytime_icon': row['daytime_icon'],
                    'rawness_percent': row['rawness_percent'],
                    'context': row['context'],
                    'artistic_description': row['artistic_description'],
                    'keywords': row['keywords'],
                    'keyword_confidences': row['keyword_confidences'],
                    'vibe': row['vibe']
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