import os
import json
import torch
import open_clip
import random
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

IMAGE_DIR = "../../images"
OUTPUT_JSON = "temperament_scores.json"

# Load model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Define comparison prompts
prompts = [
    "chaotic energetic sharp angles straight warm hot ",
    "calm quiet chill relaxed soft cool curved  gentle"
]
text_tokens = tokenizer(prompts).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

def get_temperament_score(image_path):
    try:
        # Try to open and convert image to RGB
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            image = preprocess(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                image_features = model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                similarity = (image_features @ text_features.T)[0][0].item()  # similarity to "chaotic"
                return similarity
    except Exception as e:
        logging.error(f"Failed to process {image_path}: {str(e)}")
        return float("-inf")

def find_all_images(directory):
    """Find all image files recursively in the directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
    image_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))
    
    return image_files

def main():
    # Get all image files recursively
    image_files = find_all_images(IMAGE_DIR)
    
    if not image_files:
        logging.error(f"No images found in {IMAGE_DIR}")
        return
    
    logging.info(f"Found {len(image_files)} images to process")
    scores = []

    # Process images with progress bar
    for image_path in tqdm(image_files, desc="Processing images", unit="image"):
        score = get_temperament_score(image_path)
        if score != float("-inf"):  # Only add if processing was successful
            # Get relative path from IMAGE_DIR
            rel_path = os.path.relpath(image_path, IMAGE_DIR)
            scores.append((rel_path, score))
        else:
            logging.warning(f"Skipped {image_path} due to processing error")

    if not scores:
        logging.error("No images were successfully processed")
        return

    # Sort by score ascending (lowest = 1)
    scores.sort(key=lambda x: x[1])

    # Create ranked + randomized output
    ranked = {
        filename: {
            "temperament_score": rank + 1,
            "intimacy_score": random.randint(1, 10)
        }
        for rank, (filename, _) in enumerate(scores)
    }

    # Save results
    with open(OUTPUT_JSON, "w") as f:
        json.dump(ranked, f, indent=2)

    logging.info(f"✅ Successfully processed {len(scores)} images")
    logging.info(f"✅ Results saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
