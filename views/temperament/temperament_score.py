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
import subprocess
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

IMAGE_DIR = "../../images2"  # Changed to images2
OUTPUT_JSON = "temperament_scores_images2.json"  # New output file

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

def convert_heic_to_jpeg(heic_path):
    """Convert HEIC file to JPEG using sips (macOS) or ImageMagick"""
    try:
        # Create a temporary file for the converted image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            jpeg_path = tmp_file.name
        
        # Try using sips first (macOS)
        try:
            result = subprocess.run(['sips', '-s', 'format', 'jpeg', heic_path, '--out', jpeg_path], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return jpeg_path
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Try using ImageMagick if sips fails
        try:
            result = subprocess.run(['magick', heic_path, jpeg_path], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return jpeg_path
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # If both fail, return None
        return None
    except Exception as e:
        logging.warning(f"Failed to convert HEIC {heic_path}: {str(e)}")
        return None

def get_temperament_score(image_path):
    temp_file = None
    try:
        # Handle HEIC files
        if image_path.lower().endswith(('.heic', '.heif')):
            temp_file = convert_heic_to_jpeg(image_path)
            if temp_file is None:
                return float("-inf")
            image_path = temp_file
        
        # Try to open and convert image to RGB
        with Image.open(image_path) as img:
            # Handle GIF files - take first frame
            if img.format == 'GIF':
                try:
                    img.seek(0)
                except EOFError:
                    return float("-inf")
            
            # Convert to RGB
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Verify image has content
            if img.size[0] == 0 or img.size[1] == 0:
                return float("-inf")
            
            image = preprocess(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                image_features = model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                similarity = (image_features @ text_features.T)[0][0].item()  # similarity to "chaotic"
                return similarity
                
    except (OSError, IOError, ValueError, EOFError) as e:
        logging.error(f"Failed to process {image_path}: {str(e)}")
        return float("-inf")
    except Exception as e:
        logging.error(f"Unexpected error processing {image_path}: {str(e)}")
        return float("-inf")
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except:
                pass

def find_all_images(directory):
    """Find all image files recursively in the directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.heic', '.HEIC', '.GIF', '.heif', '.HEIF'}
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
    skipped_count = 0

    # Process images with progress bar
    for image_path in tqdm(image_files, desc="Processing images", unit="image"):
        score = get_temperament_score(image_path)
        if score != float("-inf"):  # Only add if processing was successful
            # Get relative path from IMAGE_DIR
            rel_path = os.path.relpath(image_path, IMAGE_DIR)
            scores.append((rel_path, score))
        else:
            skipped_count += 1
            logging.warning(f"Skipped {image_path} due to processing error")

    if not scores:
        logging.error("No images were successfully processed")
        return

    logging.info(f"Successfully processed {len(scores)} images, skipped {skipped_count} images")

    # Sort by score ascending (lowest = 1)
    scores.sort(key=lambda x: x[1])

    # Create ranked output (removed intimacy score)
    ranked = {
        filename: {
            "temperament_score": rank + 1
        }
        for rank, (filename, _) in enumerate(scores)
    }

    # Save results
    with open(OUTPUT_JSON, "w") as f:
        json.dump(ranked, f, indent=2)

    logging.info(f"✅ Successfully processed {len(scores)} images")
    logging.info(f"✅ Skipped {skipped_count} images")
    logging.info(f"✅ Results saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()


