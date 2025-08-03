#!/usr/bin/env python3
import json
import os

# Read the list of image files
with open('images_list.txt', 'r') as f:
    image_files = [line.strip() for line in f.readlines()]

# Sort the files numerically
def natural_sort_key(filename):
    # Extract the number from filename like "image123.jpg"
    import re
    match = re.search(r'image(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0

image_files.sort(key=natural_sort_key)

# Write to images.json
with open('images/images.json', 'w') as f:
    json.dump(image_files, f, indent=2)

print(f"Generated images.json with {len(image_files)} images")
print("First 10 images:", image_files[:10])
print("Last 10 images:", image_files[-10:]) 