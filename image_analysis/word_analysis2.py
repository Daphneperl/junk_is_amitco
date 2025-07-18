import os
import json
import torch
from PIL import Image
import numpy as np
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel
import re
from typing import List, Dict, Tuple
import logging
from tqdm import tqdm
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArtisticImageAnalyzer:
    def __init__(self, images_folder: str, output_file: str = "artistic_analysis.json"):
        """
        Initialize the Artistic Image Analyzer
        
        Args:
            images_folder: Path to folder containing images
            output_file: Output JSON file path
        """
        self.images_folder = images_folder
        self.output_file = output_file
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load reference words for keyword filtering
        self.reference_words = self._load_reference_words()
        
        # Initialize models
        self._load_models()
        
        # Vibe categories for mood detection
        self.vibe_categories = [
            "melancholic", "surreal", "solemn", "sharp", "whimsical", "dark", 
            "ethereal", "raw", "dreamy", "intense", "serene", "chaotic", 
            "mysterious", "playful", "somber", "vibrant", "minimal", "complex",
            "organic", "geometric", "fluid", "rigid", "warm", "cool", "neutral"
        ]
        
        # Keywords to avoid (medium/context words)
        self.avoid_keywords = {
            "notebook", "scan", "drawing", "photo", "page", "cover", "illustration",
            "image", "picture", "artwork", "painting", "sketch", "text", "document",
            "paper", "canvas", "medium", "style", "technique", "composition", "notebooks", "photos", "scans", "covers", "illustrations","drawings",
            "paintings", "art", "sketches","draft", "drafts", "pencil", "journal", "guestbook", "protfolio", "envelope", "words",
        }
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

    def _load_reference_words(self) -> set:
        """Load reference words from the provided text file"""
        try:
            with open("views/rhizome/google-10000-english-gpt-clean.txt", "r") as f:
                words = set(line.strip().lower() for line in f if line.strip())
            logger.info(f"Loaded {len(words)} reference words")
            return words
        except FileNotFoundError:
            logger.warning("Reference word file not found, using empty set")
            return set()

    def _load_models(self):
        """Load BLIP-2 and CLIP models"""
        logger.info("Loading BLIP-2 model...")
        self.blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        logger.info("Loading CLIP model...")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        
        logger.info("Models loaded successfully")

    def _get_image_files(self) -> List[str]:
        """Get all image files from the images folder"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
        image_files = []
        
        for filename in os.listdir(self.images_folder):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(self.images_folder, filename))
        
        logger.info(f"Found {len(image_files)} image files")
        return sorted(image_files)

    def _generate_description(self, image: Image.Image) -> str:
        """Generate a vivid, artistic description using BLIP-2, avoiding medium/context words in the output."""
        medium_words = {"page", "paper", "watercolor", "drawing","background", "sketch", "painting", "canvas", "piece", "notebook", "document", "text", "scan", "piece of paper", "photo", "image", "picture", "artwork", "illustration"}
        try:
            # Try multiple generation strategies with different parameters
            generation_configs = [
                # Strategy 1: Direct visual description
                {
                    "prompt": "A visual composition showing",
                    "max_new_tokens": 80,
                    "temperature": 0.7,
                    "num_beams": 5,
                    "top_p": 0.9
                },
                # Strategy 2: Natural visual prompt, moderate parameters
                {
                    "prompt": "This image contains",
                    "max_new_tokens": 70,
                    "temperature": 0.8,
                    "num_beams": 4,
                    "top_p": 0.85
                },
                # Strategy 3: Minimal prompt
                {
                    "prompt": "Visual elements include",
                    "max_new_tokens": 60,
                    "temperature": 0.6,
                    "num_beams": 3,
                    "top_p": 0.8
                }
            ]
            
            for config in generation_configs:
                if config["prompt"]:
                    inputs = self.blip_processor(image, text=config["prompt"], return_tensors="pt").to(self.device)
                else:
                    inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    generated_ids = self.blip_model.generate(
                        **inputs,
                        max_new_tokens=config["max_new_tokens"],
                        num_beams=config["num_beams"],
                        temperature=config["temperature"],
                        do_sample=True,
                        top_p=config["top_p"],
                        repetition_penalty=1.5,
                        no_repeat_ngram_size=2
                    )
                
                description = self.blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                
                # Clean up the description
                description = re.sub(r'^.*?:', '', description).strip()
                description = re.sub(r'https?://\S+', '', description)
                description = re.sub(r'www\.\S+', '', description)
                
                # Remove text-based content (likely OCR artifacts) - be more selective
                text_patterns = [
                    r'\b(original art print by|signed|dated|copyright|©|™|®)\b',
                    r'\b\d{4}\b',  # Years
                ]
                for pattern in text_patterns:
                    description = re.sub(pattern, '', description, flags=re.IGNORECASE)
                
                # Clean up extra whitespace and punctuation
                description = re.sub(r'\s+', ' ', description).strip()
                description = re.sub(r'^[^\w]*', '', description)  # Remove leading non-word chars
                description = re.sub(r'[^\w]*$', '', description)  # Remove trailing non-word chars
                
                # Remove common prompt phrases (both start and end)
                prompt_phrases = [
                    r'^a visual composition showing[\s\.]*',
                    r'^this image contains[\s\.]*',
                    r'^visual elements include[\s\.]*',
                    r'^describe what you see[\s\.]*',
                    r'^what do you see[\s\.]*',
                    r'^describe the visual elements[\s\.]*',
                    r'^this is[\s\.]*',
                    r'^the image shows[\s\.]*',
                    r'^whats in this[\s\.]*',
                    r'^tell me about this[\s\.]*',
                    r'[\s\.]*colors, shapes, figures, and composition in detail[\s\.]*$',
                    r'[\s\.]*visual elements, colors, shapes, figures, and composition in detail[\s\.]*$'
                ]
                for phrase in prompt_phrases:
                    description = re.sub(phrase, '', description, flags=re.IGNORECASE).strip()
                
                # Remove repetitive phrases
                words = description.split()
                if len(words) > 3:
                    cleaned_words = []
                    for i, word in enumerate(words):
                        if i == 0 or word.lower() != words[i-1].lower():
                            cleaned_words.append(word)
                    description = ' '.join(cleaned_words)
                
                # Check for medium words and quality
                description_lower = description.lower()
                has_medium_words = any(word in description_lower for word in medium_words)
                
                # Quality check: description should be detailed enough and not contain text artifacts
                is_too_generic = len(description.split()) < 3 or description.lower() in ["black and white", "colorful", "abstract", "artistic"]
                has_text_artifacts = any(word in description_lower for word in ["original art print by", "signed", "dated", "copyright"])
                is_incomplete = description.startswith("-") or description.startswith("a of") or len(description.strip()) < 8
                
                if (not has_medium_words and 
                    not has_text_artifacts and 
                    not is_too_generic and 
                    not is_incomplete and
                    8 <= len(description) <= 200):
                    return description
            
            # If all strategies failed, try filtering medium words from the best attempt
            if len(description) > 0:
                filtered_words = [w for w in words if all(mw not in w.lower() for mw in medium_words)]
                filtered_description = ' '.join(filtered_words)
                if 10 <= len(filtered_description) <= 200:
                    return filtered_description

            # Final fallback
            return "A distinctive visual composition with unique artistic elements and compelling visual qualities."
        except Exception as e:
            logger.error(f"Error generating description: {e}")
            return "A distinctive visual composition with unique artistic elements and compelling visual qualities."

    def _extract_nouns_from_description(self, description: str, image_features=None) -> List[Dict]:
        """Extract nouns from BLIP description and return with confidence scores"""
        try:
            # Tokenize and tag parts of speech
            tokens = word_tokenize(description.lower())
            pos_tags = pos_tag(tokens)
            
            # Extract nouns (NN, NNS, NNP, NNPS)
            nouns = []
            for word, tag in pos_tags:
                if tag.startswith('NN') and len(word) >= 3:
                    # Clean the word
                    word = re.sub(r'[^\w]', '', word)
                    if word and self._is_valid_keyword(word):
                        # Calculate CLIP similarity score if image_features provided
                        if image_features is not None:
                            try:
                                # Get text features for this noun
                                text_inputs = self.clip_processor(text=[word], return_tensors="pt").to(self.device)
                                with torch.no_grad():
                                    text_features = self.clip_model.get_text_features(**text_inputs)
                                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                                
                                # Calculate similarity
                                similarity = torch.matmul(image_features, text_features.T).squeeze().item()
                                # Add 0.5 to the original score
                                confidence = similarity + 0.5
                            except Exception as e:
                                logger.warning(f"Error calculating similarity for noun '{word}': {e}")
                                confidence = 0.5  # Fallback to original behavior
                        else:
                            confidence = 0.5  # Fallback when no image_features provided
                        
                        nouns.append({"keyword": word, "confidence": round(confidence, 4)})
            
            # Remove duplicates while preserving order
            seen = set()
            unique_nouns = []
            for noun in nouns:
                if noun["keyword"] not in seen:
                    seen.add(noun["keyword"])
                    unique_nouns.append(noun)
            
            return unique_nouns[:5]  # Return top 5 nouns
            
        except Exception as e:
            logger.error(f"Error extracting nouns from description: {e}")
            return []

    def _extract_keywords(self, image: Image.Image, description: str = "") -> List[Dict]:
        """Extract high-confidence keywords using CLIP and nouns from description"""
        try:
            # Prepare image for CLIP
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            
            # Get image features
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Use reference words as potential keywords, filtered for relevance
            if not self.reference_words:
                logger.warning("No reference words available, using fallback keywords")
                potential_keywords = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "black", "white", "gray"]
            else:
                # Filter reference words to exclude avoid keywords and very short words
                potential_keywords = []
                for word in self.reference_words:
                    word_lower = word.lower()
                    if (len(word) >= 3 and 
                        word_lower not in self.avoid_keywords and
                        not word_lower.isdigit() and
                        not any(char.isdigit() for char in word)):
                        potential_keywords.append(word)
                
                logger.info(f"Using {len(potential_keywords)} reference words for CLIP analysis")
            
            # Process keywords in batches to avoid memory issues
            batch_size = 100
            all_similarities = []
            
            for i in range(0, len(potential_keywords), batch_size):
                batch_keywords = potential_keywords[i:i + batch_size]
                
                # Get text features for batch
                text_inputs = self.clip_processor(text=batch_keywords, return_tensors="pt", padding=True, truncation=True).to(self.device)
                
                with torch.no_grad():
                    text_features = self.clip_model.get_text_features(**text_inputs)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarities for this batch
                batch_similarities = torch.matmul(image_features, text_features.T).squeeze()
                all_similarities.append(batch_similarities)
            
            # Combine all similarities
            similarities = torch.cat(all_similarities)
            
            # Get the actual keywords with their scores
            keyword_scores = [(potential_keywords[i], similarities[i].item()) for i in range(len(similarities))]
            
            # Sort by score
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Filter keywords before selecting top 8
            filtered_keyword_scores = []
            for kw, score in keyword_scores:
                if score > 0.3:  # Apply confidence threshold
                    filtered_keyword_scores.append((kw, score))
            
            # If no keywords meet threshold, get top 20 and filter
            if not filtered_keyword_scores:
                top_20_scores = keyword_scores[:20]
                filtered_keyword_scores = [(kw, score) for kw, score in top_20_scores]
            
            # Apply additional filtering
            final_keywords = []
            for kw, score in filtered_keyword_scores:
                if self._is_valid_keyword(kw):
                    final_keywords.append({"keyword": kw, "confidence": round(score, 4)})
                    if len(final_keywords) >= 8:  # Stop when we have 8 valid keywords
                        break
            
            # Extract nouns from description and add them
            if description:
                description_nouns = self._extract_nouns_from_description(description, image_features)
                for noun in description_nouns:
                    # Check if noun already exists in final_keywords
                    existing_keywords = [kw["keyword"] for kw in final_keywords]
                    if noun["keyword"] not in existing_keywords:
                        final_keywords.append(noun)
                        if len(final_keywords) >= 12:  # Allow up to 12 total keywords
                            break
            
            # Sort keywords by confidence score (highest first)
            final_keywords.sort(key=lambda x: x["confidence"], reverse=True)
            
            return final_keywords[:12]  # Return up to 12 keywords (8 CLIP + up to 4 nouns)
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return [
                {"keyword": "visual", "confidence": 0.5},
                {"keyword": "composition", "confidence": 0.4},
                {"keyword": "artistic", "confidence": 0.3}
            ]

    def _is_valid_keyword(self, keyword: str) -> bool:
        """Check if a keyword is valid based on filtering criteria"""
        keyword_lower = keyword.lower()
        
        # Skip if in avoid list
        if keyword_lower in self.avoid_keywords:
            return False
            
        # Skip very short words
        if len(keyword) < 3:
            return False
            
        # Skip words with numbers
        if any(char.isdigit() for char in keyword):
            return False
            
        return True

    def _filter_keywords(self, keywords: List[str]) -> List[str]:
        """Filter keywords based on avoid list and quality criteria"""
        filtered = []
        
        for keyword in keywords:
            if self._is_valid_keyword(keyword):
                filtered.append(keyword)
        
        return filtered

    def _detect_vibe(self, description: str, keywords: List[Dict]) -> str:
        """Detect the overall vibe/mood of the image"""
        # Extract just the keyword strings for vibe analysis
        keyword_strings = [kw["keyword"] for kw in keywords]
        # Combine description and keywords for vibe analysis
        text_for_vibe = f"{description} {' '.join(keyword_strings)}".lower()
        
        # Define vibe indicators
        vibe_indicators = {
            "melancholic": ["dark", "sad", "blue", "gray", "somber", "quiet", "gentle", "soft"],
            "surreal": ["dreamy", "ethereal", "mysterious", "strange", "unusual", "fantastical"],
            "solemn": ["serious", "dignified", "formal", "respectful", "quiet", "calm"],
            "sharp": ["angular", "geometric", "bold", "contrast", "dramatic", "intense"],
            "whimsical": ["playful", "fun", "bright", "colorful", "cheerful", "light"],
            "dark": ["black", "shadow", "gloomy", "mysterious", "intense", "dramatic"],
            "ethereal": ["light", "transparent", "floating", "delicate", "soft", "dreamy"],
            "raw": ["rough", "textured", "natural", "organic", "unrefined", "earthy"],
            "dreamy": ["soft", "blurred", "gentle", "peaceful", "calm", "serene"],
            "intense": ["bold", "dramatic", "powerful", "strong", "vibrant", "energetic"],
            "serene": ["calm", "peaceful", "quiet", "gentle", "soft", "tranquil"],
            "chaotic": ["dynamic", "energetic", "random", "complex", "busy", "active"],
            "mysterious": ["dark", "shadow", "unknown", "hidden", "obscure", "enigmatic"],
            "playful": ["bright", "colorful", "fun", "light", "cheerful", "energetic"],
            "somber": ["dark", "serious", "quiet", "calm", "gentle", "soft"],
            "vibrant": ["bright", "colorful", "energetic", "dynamic", "bold", "intense"],
            "minimal": ["simple", "clean", "sparse", "quiet", "calm", "gentle"],
            "complex": ["detailed", "intricate", "layered", "dynamic", "rich", "full"],
            "organic": ["natural", "flowing", "curved", "soft", "gentle", "fluid"],
            "geometric": ["angular", "sharp", "structured", "organized", "precise"],
            "fluid": ["flowing", "smooth", "gentle", "soft", "organic", "natural"],
            "rigid": ["angular", "sharp", "structured", "organized", "precise"],
            "warm": ["red", "orange", "yellow", "brown", "gold", "cozy"],
            "cool": ["blue", "green", "purple", "teal", "silver", "calm"],
            "neutral": ["gray", "beige", "white", "black", "simple", "quiet"]
        }
        
        # Calculate scores for each vibe
        vibe_scores = {}
        for vibe, indicators in vibe_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_for_vibe)
            vibe_scores[vibe] = score
        
        # Return the vibe with highest score, or default
        if max(vibe_scores.values()) > 0:
            return max(vibe_scores, key=vibe_scores.get)
        else:
            return random.choice(["neutral", "serene", "mysterious"])

    def analyze_image(self, image_path: str) -> Dict:
        """Analyze a single image and return results"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Generate description
            description = self._generate_description(image)
            
            # Extract keywords (pass description to include nouns)
            keywords = self._extract_keywords(image, description)
            
            # Ensure keywords are sorted by confidence score (highest first)
            keywords.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Detect vibe
            vibe = self._detect_vibe(description, keywords)
            
            # Get filename without extension
            filename = os.path.splitext(os.path.basename(image_path))[0]
            
            return {
                "filename": filename,
                "description": description,
                "keywords": keywords,
                "vibe": vibe
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {image_path}: {e}")
            fallback_keywords = [
                {"keyword": "visual", "confidence": 0.5},
                {"keyword": "composition", "confidence": 0.4},
                {"keyword": "artistic", "confidence": 0.3}
            ]
            # Sort fallback keywords by confidence score (highest first)
            fallback_keywords.sort(key=lambda x: x["confidence"], reverse=True)
            
            return {
                "filename": os.path.splitext(os.path.basename(image_path))[0],
                "description": "An intriguing visual composition with distinctive artistic elements.",
                "keywords": fallback_keywords,
                "vibe": "mysterious"
            }

    def analyze_all_images(self):
        """Analyze all images in the folder and save results to JSON"""
        image_files = self._get_image_files()
        
        if not image_files:
            logger.error("No image files found!")
            return
        
        results = []
        
        logger.info(f"Starting analysis of {len(image_files)} images...")
        
        for image_path in tqdm(image_files, desc="Analyzing images"):
            result = self.analyze_image(image_path)
            results.append(result)
            
            # Log progress every 50 images
            if len(results) % 50 == 0:
                logger.info(f"Processed {len(results)} images...")
        
        # Save results to JSON
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analysis complete! Results saved to {self.output_file}")
        logger.info(f"Processed {len(results)} images successfully")

def main():
    """Main function to run the analysis"""
    # Configuration
    images_folder = "images"  # Path to main images folder
    output_file = "image_analysis/artistic_analysis.json"  # Output file name for main analysis
    
    # Check if images folder exists
    if not os.path.exists(images_folder):
        logger.error(f"Images folder '{images_folder}' not found!")
        return
    
    # Create analyzer and run analysis
    analyzer = ArtisticImageAnalyzer(images_folder, output_file)
    analyzer.analyze_all_images()

if __name__ == "__main__":
    main()
