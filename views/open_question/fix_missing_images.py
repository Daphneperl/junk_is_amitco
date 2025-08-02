import json
import os

def generate_cat_scores_for_existing_images():
    """Generate scores only for images that actually exist"""
    scores = {}
    scoring_word = "cat"
    
    # Get the list of existing images
    images_dir = "../../images/"
    if os.path.exists(images_dir):
        image_files = []
        for filename in os.listdir(images_dir):
            if filename.startswith("image") and any(filename.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                image_files.append(filename)
        
        # Sort by image number
        image_files.sort(key=lambda x: int(x.replace('image', '').split('.')[0]) if x.replace('image', '').split('.')[0].isdigit() else 0)
        
        print(f"Found {len(image_files)} existing images")
        
        # Generate scores only for existing images
        for filename in image_files:
            score_hash = hash(filename + scoring_word) % 100 + 1
            scores[filename] = score_hash
        
        # Save to file
        with open('image_scores_openQ.json', 'w') as f:
            json.dump(scores, f, indent=2)
        
        print(f"Generated scores for {len(scores)} images")
        print("First 10 scores:")
        for i, (filename, score) in enumerate(list(scores.items())[:10]):
            print(f"  {filename}: {score}")
    else:
        print(f"Images directory not found: {images_dir}")

if __name__ == "__main__":
    generate_cat_scores_for_existing_images() 