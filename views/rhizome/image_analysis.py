from PIL import Image
import json
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import os

# Define a rich set of themes and concepts that are relevant for artistic analysis
THEMES = [
    # Emotional themes
    "joy", "sadness", "anger", "peace", "loneliness", "love", "conflict", "harmony",
    # Abstract concepts
    "nature", "technology", "tradition", "modernity", "chaos", "order", "freedom", "constraint",
    # Visual elements
    "geometric", "organic", "symmetrical", "asymmetrical", "minimal", "complex",
    # Cultural themes
    "identity", "memory", "power", "society", "spirituality", "mythology", "urban life", "rural life",
    # Artistic movements
    "surreal", "abstract", "figurative", "conceptual", "minimalist", "expressionist",
    # Compositional elements
    "dynamic", "static", "balanced", "contrasting", "rhythmic", "flowing",
    # Emotional atmospheres
    "mysterious", "playful", "serene", "dramatic", "intimate", "monumental",
    # Contemporary themes
    "digital culture", "environmental", "social justice", "globalization", "isolation", "connection"
]

def load_keywords():
    # Get the path to the keywords file
    keywords_file = Path(__file__).parent / "google-10000-english-usa.txt"
    
    with open(keywords_file, 'r') as f:
        # Filter out very short words and common stop words
        stop_words = {'the', 'of', 'and', 'to', 'a', 'in', 'for', 'is', 'on', 'that', 'with', 'at', 'by', 'this'}
        words = [word.strip() for word in f.readlines()]
        # Return all words that are longer than 2 characters and not stop words
        return [word for word in words if len(word) > 2 and word not in stop_words]

def analyze_images():
    # Initialize CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Load keywords
    keywords = load_keywords()
    
    # Get the absolute path to the project root directory
    project_root = Path(__file__).resolve().parents[2]
    
    # Get the path to the images directory
    images_dir = project_root / "images"
    
    # Dictionary to store results
    image_keywords = {}
    
    # Get all image files
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + \
                 list(images_dir.glob("*.png")) + list(images_dir.glob("*.gif")) + \
                 list(images_dir.glob("*.webp"))
    
    # Process each image
    for img_path in tqdm(image_files, desc="Analyzing images"):
        try:
            # Open and process the image
            image = Image.open(img_path)
            
            # Convert RGBA to RGB if necessary
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            # First analyze themes
            text_inputs = processor(
                text=THEMES,
                return_tensors="pt",
                padding=True
            ).to(device)
            
            image_inputs = processor(
                images=image,
                return_tensors="pt"
            ).to(device)
            
            # Get theme similarity scores
            with torch.no_grad():
                outputs = model(**{**image_inputs, **text_inputs})
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds
                
                theme_similarity = torch.nn.functional.cosine_similarity(
                    image_features[:, None],
                    text_features[None, :],
                    dim=-1
                )
            
            # Always get exactly 2 top themes, regardless of confidence
            top_theme_scores, top_theme_indices = theme_similarity[0].topk(2)
            
            # Extract exactly 2 themes
            themes = [
                {
                    "theme": THEMES[idx],
                    "confidence": float(score.cpu().numpy())
                }
                for score, idx in zip(top_theme_scores, top_theme_indices)
            ]
            
            # Then analyze keywords from Google's word list
            text_inputs = processor(
                text=keywords,
                return_tensors="pt",
                padding=True
            ).to(device)
            
            # Get keyword similarity scores
            with torch.no_grad():
                outputs = model(**{**image_inputs, **text_inputs})
                text_features = outputs.text_embeds
                
                keyword_similarity = torch.nn.functional.cosine_similarity(
                    image_features[:, None],
                    text_features[None, :],
                    dim=-1
                )
            
            # Always get exactly 8 top keywords, regardless of confidence
            top_keyword_scores, top_keyword_indices = keyword_similarity[0].topk(8)
            
            # Extract exactly 8 keywords
            keywords_list = [
                {
                    "keyword": keywords[idx],
                    "confidence": float(score.cpu().numpy())
                }
                for score, idx in zip(top_keyword_scores, top_keyword_indices)
            ]
            
            # Store in our dictionary using relative path for the key
            rel_path = str(img_path.relative_to(images_dir))
            image_keywords[rel_path] = {
                "themes": themes,  # Exactly 2 theme words
                "keywords": keywords_list  # Exactly 8 Google words
            }
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue
            
    # Save results to JSON file in the same directory as this script
    output_path = Path(__file__).parent / "image_keywords.json"
    with open(output_path, "w") as f:
        json.dump(image_keywords, f, indent=2)
    
    print(f"\nAnalysis complete! Results saved to {output_path}")

if __name__ == "__main__":
    analyze_images() 