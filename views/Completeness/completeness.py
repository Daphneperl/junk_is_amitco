import os
import json
from PIL import Image
from tqdm import tqdm
import torch
from transformers import CLIPProcessor, CLIPModel
import tempfile
import shutil

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
image_folder = os.path.join(project_root, "images2")

# --- Convert HEIC to JPEG and get all image files ---
def convert_heic_to_jpeg(heic_path):
    """Convert HEIC file to JPEG format"""
    try:
        with Image.open(heic_path) as img:
            # Create temporary file for JPEG
            temp_jpeg = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            img.convert('RGB').save(temp_jpeg.name, 'JPEG', quality=95)
            return temp_jpeg.name
    except Exception as e:
        print(f"Error converting {heic_path}: {e}")
        return None

def get_image_files_with_conversion():
    """Get all image files, converting HEIC to JPEG as needed"""
    image_files = []
    temp_files = []
    
    for f in os.listdir(image_folder):
        f_lower = f.lower()
        if f_lower.endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
            image_files.append(f)
        elif f_lower.endswith((".heic", ".HEIC")):
            # Convert HEIC to JPEG
            heic_path = os.path.join(image_folder, f)
            jpeg_path = convert_heic_to_jpeg(heic_path)
            if jpeg_path:
                # Create a temporary mapping for the converted file
                temp_files.append((f, jpeg_path))
                image_files.append(f)  # Keep original filename for output
    
    return image_files, temp_files

# --- Get similarity to sketch prompt ---
def get_sketch_similarity(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, **text_inputs)
            similarity = outputs.logits_per_image.squeeze().item()
        return similarity
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# --- Get image files and convert HEIC files ---
print("🔄 Scanning images2 folder and converting HEIC files...")
image_files, temp_files = get_image_files_with_conversion()

# Create mapping for converted files
converted_paths = {original: temp_path for original, temp_path in temp_files}

print(f"📁 Found {len(image_files)} image files")
if temp_files:
    print(f"🔄 Converted {len(temp_files)} HEIC files to JPEG")

# --- Score all images ---
print("🔄 Scoring images inversely to their sketch-likeness...")
similarities = {}
failed_files = []

for fname in tqdm(image_files):
    # Use converted path if available, otherwise use original path
    if fname in converted_paths:
        path = converted_paths[fname]
    else:
        path = os.path.join(image_folder, fname)
    
    score = get_sketch_similarity(path)
    if score is not None:
        similarities[fname] = score
    else:
        failed_files.append(fname)

# Clean up temporary files
for original, temp_path in temp_files:
    try:
        os.unlink(temp_path)
    except:
        pass

if failed_files:
    print(f"⚠️ Failed to process {len(failed_files)} files: {failed_files}")

if not similarities:
    print("❌ No images were successfully processed!")
    exit(1)

# --- Invert the similarity: lower similarity → higher score
min_sim = min(similarities.values())
max_sim = max(similarities.values())

# Inverted: most sketch-like (high sim) = score 1; least sketch-like = score 500
final_scores = {
    fname: int(((max_sim - sim) / (max_sim - min_sim)) * 499 + 1)
    for fname, sim in similarities.items()
}

# --- Save JSON in the Completeness directory ---
output_path = os.path.join(script_dir, "inverted_sketchiness_scores_images2.json")
with open(output_path, "w") as f:
    json.dump(final_scores, f, indent=2)

print(f"✅ Done. Processed {len(similarities)} images.")
print(f"📊 Score range: {min(final_scores.values())} - {max(final_scores.values())}")
print(f"💾 Saved to '{output_path}'")
