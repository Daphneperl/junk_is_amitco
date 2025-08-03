#!/usr/bin/env python3
import json
import os

def get_image_extension(filename):
    """Get the actual file extension for an image"""
    base_name = filename.split('.')[0]  # Remove any existing extension
    possible_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.HEIC', '.heic']
    
    for ext in possible_extensions:
        full_path = f"images/{base_name}{ext}"
        if os.path.exists(full_path):
            return ext
    
    # Default to .jpg if not found
    return '.jpg'

def update_artistic_data():
    """Update artistic data to include correct file extensions"""
    with open('image_analysis/artistic_analysis_filtered.json', 'r') as f:
        data = json.load(f)
    
    for item in data:
        filename = item['filename']
        extension = get_image_extension(filename)
        item['filename_with_extension'] = f"{filename}{extension}"
    
    with open('image_analysis/artistic_analysis_filtered.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Updated {len(data)} image records with correct extensions")

if __name__ == "__main__":
    update_artistic_data() 