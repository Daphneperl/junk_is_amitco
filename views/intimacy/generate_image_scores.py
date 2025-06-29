import os
import json
import random
from pathlib import Path

def generate_image_scores():
    """
    Generate a JSON file with random scores (0-10) for each image in the images folder.
    """
    # Get the path to the images folder (relative to the project root)
    project_root = Path(__file__).parent.parent.parent
    images_folder = project_root / "images"
    
    # Get all image files from the images folder
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
    image_files = []
    
    if images_folder.exists():
        for file_path in images_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(file_path.name)
    
    # Sort the image files for consistent ordering
    image_files.sort()
    
    # Generate random scores for each image
    image_scores = {}
    for image_file in image_files:
        # Generate a random integer from 0 to 10
        score = random.randint(0, 10)
        image_scores[image_file] = score
    
    # Create the output file path in the intimacy folder
    output_file = Path(__file__).parent / "image_scores.json"
    
    # Write the scores to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(image_scores, f, indent=2, ensure_ascii=False)
    
    print(f"Generated scores for {len(image_files)} images")
    print(f"Output saved to: {output_file}")
    
    # Print a sample of the generated scores
    print("\nSample scores:")
    sample_items = list(image_scores.items())[:10]
    for image_name, score in sample_items:
        print(f"  {image_name}: {score}")
    
    if len(image_scores) > 10:
        print(f"  ... and {len(image_scores) - 10} more images")

if __name__ == "__main__":
    # Set a seed for reproducible results (optional - remove for truly random)
    # random.seed(42)
    
    generate_image_scores() 