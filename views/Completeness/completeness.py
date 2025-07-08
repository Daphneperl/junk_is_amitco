import os
import json
from PIL import Image
from tqdm import tqdm
import torch
from transformers import CLIPProcessor, CLIPModel

# --- Load CLIP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# --- Prompt: sketch-like ---
prompt = "a rough sketch draft or doodle on notebook paper"
text_inputs = processor(text=[prompt], return_tensors="pt", padding=True).to(device)

# --- Image folder (relative to project root) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))  # Go up two levels from Completeness/
image_folder = os.path.join(project_root, "images")
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif"))]

# --- Get similarity to sketch prompt ---
def get_sketch_similarity(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, **text_inputs)
        similarity = outputs.logits_per_image.squeeze().item()
    return similarity

# --- Score all images ---
print("🔄 Scoring images inversely to their sketch-likeness...")
similarities = {}
for fname in tqdm(image_files):
    path = os.path.join(image_folder, fname)
    score = get_sketch_similarity(path)
    similarities[fname] = score

# --- Invert the similarity: lower similarity → higher score
min_sim = min(similarities.values())
max_sim = max(similarities.values())

# Inverted: most sketch-like (high sim) = score 1; least sketch-like = score 500
final_scores = {
    fname: int(((max_sim - sim) / (max_sim - min_sim)) * 499 + 1)
    for fname, sim in similarities.items()
}

# --- Save JSON in the Completeness directory ---
output_path = os.path.join(script_dir, "inverted_sketchiness_scores.json")
with open(output_path, "w") as f:
    json.dump(final_scores, f, indent=2)

print(f"✅ Done. Saved to '{output_path}'")
