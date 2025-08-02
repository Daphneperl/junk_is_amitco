from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import random
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variable to store current scores
current_scores = {}

# Global variable for the target word - easily changeable
TARGET_WORD = "animal"  # Change this to any word you want to search for

def generate_semantic_scores(query, image_files=None):
    """Generate semantic scores for images based on similarity to the target word"""
    import json
    import os
    
    def load_artistic_analysis():
        """Load the artistic analysis data"""
        analysis_file = "../../image_analysis/artistic_analysis_filtered.json"
        if os.path.exists(analysis_file):
            with open(analysis_file, 'r') as f:
                return json.load(f)
        else:
            print(f"Artistic analysis file not found: {analysis_file}")
            return []
    
    def calculate_semantic_similarity(image_data, target_word=TARGET_WORD):
        """Calculate semantic similarity to the target word based on keywords and descriptions"""
        score = 0.0
        
        # Check if the target word appears in the description
        description = image_data.get('description', '').lower()
        if target_word in description:
            score += 50.0  # High score for direct mention in description
        
        # Check keywords for exact matches and semantic relationships
        keywords = image_data.get('keywords', [])
        for keyword_data in keywords:
            keyword = keyword_data.get('keyword', '').lower()
            confidence = keyword_data.get('confidence', 0.0)
            
            # Exact match gets high score
            if keyword == target_word:
                score += confidence * 100.0
            elif keyword == target_word + 's':  # Plural form
                score += confidence * 80.0
            # Related words get partial scores
            elif target_word in keyword or keyword in target_word:
                score += confidence * 30.0
            
            # Target-specific semantic relationships
            if target_word == "animal":
                # Animal-related concepts
                animal_related = ['pet', 'wild', 'domestic', 'mammal', 'bird', 'fish', 'reptile', 'amphibian',
                                'dog', 'cat', 'horse', 'cow', 'pig', 'sheep', 'goat', 'chicken', 'duck',
                                'lion', 'tiger', 'elephant', 'giraffe', 'zebra', 'bear', 'wolf', 'fox',
                                'rabbit', 'mouse', 'rat', 'hamster', 'guinea pig', 'ferret', 'snake', 'lizard',
                                'turtle', 'frog', 'toad', 'fish', 'shark', 'whale', 'dolphin', 'seal',
                                'penguin', 'owl', 'eagle', 'hawk', 'parrot', 'canary', 'finch', 'sparrow',
                                'fur', 'feather', 'scale', 'paw', 'claw', 'beak', 'tail', 'wing', 'horn',
                                'antler', 'mane', 'whisker', 'snout', 'trunk', 'tusk', 'hoof', 'fleece',
                                'zoo', 'farm', 'barn', 'pasture', 'jungle', 'savanna', 'forest', 'ocean',
                                'cute', 'fierce', 'gentle', 'playful', 'sleeping', 'hunting', 'grazing',
                                'flying', 'swimming', 'running', 'jumping', 'climbing', 'crawling']
                if keyword in animal_related:
                    score += confidence * 25.0
                    
            elif target_word == "water":
                # Water-related concepts
                water_related = ['ocean', 'sea', 'river', 'lake', 'stream', 'pond', 'beach', 'swimming', 
                               'fish', 'boat', 'ship', 'rain', 'drop', 'liquid', 'blue', 'wet', 'flow',
                               'bubble', 'wave', 'splash', 'dive', 'sail', 'fishing', 'aquatic', 'marine']
                if keyword in water_related:
                    score += confidence * 25.0
                    
            elif target_word == "woman":
                # Woman-related concepts
                woman_related = ['female', 'girl', 'lady', 'person', 'human', 'face', 'portrait', 'figure',
                               'dress', 'hair', 'beauty', 'fashion', 'model', 'actress', 'mother', 'sister']
                if keyword in woman_related:
                    score += confidence * 25.0
                    
            elif target_word == "star":
                # Star-related concepts
                star_related = ['night', 'sky', 'space', 'galaxy', 'universe', 'planet', 'moon', 'sun',
                              'cosmic', 'astronomical', 'celestial', 'twinkle', 'bright', 'light', 'glow']
                if keyword in star_related:
                    score += confidence * 25.0
                    
            elif target_word == "tree":
                # Tree-related concepts
                tree_related = ['forest', 'wood', 'leaf', 'branch', 'nature', 'green', 'plant', 'garden',
                              'park', 'outdoor', 'landscape', 'natural', 'organic', 'growth', 'shade']
                if keyword in tree_related:
                    score += confidence * 25.0
                    
            elif target_word == "cat":
                # Cat-related concepts
                cat_related = ['animal', 'pet', 'feline', 'kitten', 'kitty', 'paw', 'tail', 'whisker',
                             'fur', 'domestic', 'mammal', 'cute', 'playful', 'sleeping', 'hunting']
                if keyword in cat_related:
                    score += confidence * 25.0
        
        # Add some randomness based on confidence to create more granular scores
        # This prevents all low-scoring images from getting the same score
        if score < 10:
            # For low-scoring images, add small variations based on confidence
            total_confidence = sum(k.get('confidence', 0) for k in keywords)
            score += (total_confidence * 0.1) % 5  # Small variation 0-5
        
        # Normalize score to 1-100 range
        return min(100, max(1, int(score)))
    
    # Load artistic analysis
    analysis_data = load_artistic_analysis()
    if not analysis_data:
        print("No artistic analysis data found, using fallback scoring")
        return {}
    
    # Create filename to data mapping
    image_data_map = {}
    for item in analysis_data:
        filename = item.get('filename', '')
        # Add file extension if missing
        if not any(filename.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
            # Try to find the actual file extension
            for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                if os.path.exists(f"../../images/{filename}{ext}"):
                    filename = filename + ext
                    break
        image_data_map[filename] = item
    
    # If no specific image files provided, get the list of existing images
    if image_files is None:
        images_dir = "../../images/"
        if os.path.exists(images_dir):
            image_files = []
            for filename in os.listdir(images_dir):
                if filename.startswith("image") and any(filename.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                    image_files.append(filename)
            image_files.sort(key=lambda x: int(x.replace('image', '').split('.')[0]) if x.replace('image', '').split('.')[0].isdigit() else 0)
    
    # Generate semantic scores
    scores = {}
    if image_files:
        for filename in image_files:
            if filename in image_data_map:
                # Calculate semantic similarity to the target word
                score = calculate_semantic_similarity(image_data_map[filename], query)
                scores[filename] = score
            else:
                # Default score for images without analysis data
                scores[filename] = 1
    
    return scores

@app.route('/search', methods=['POST'])
def search():
    """Handle search queries and return updated scores"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        print(f"[{datetime.now()}] Received search query: '{query}'")
        
        # Generate new scores based on query
        global current_scores
        current_scores = generate_semantic_scores(query)
        
        # Save scores to JSON file (async to avoid blocking)
        try:
            with open('image_scores.json', 'w') as f:
                json.dump(current_scores, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save to file: {e}")
        
        print(f"[{datetime.now()}] Generated {len(current_scores)} scores for query: '{query}'")
        
        return jsonify({
            'success': True,
            'message': f'Query "{query}" processed successfully',
            'scores': current_scores,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"[{datetime.now()}] Error processing search: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/scores', methods=['GET'])
def get_scores():
    """Get current scores"""
    try:
        global current_scores
        if not current_scores:
            # Load from file if exists, otherwise generate default
            if os.path.exists('image_scores.json'):
                with open('image_scores.json', 'r') as f:
                    current_scores = json.load(f)
            else:
                current_scores = generate_semantic_scores("default")
        
        return jsonify({
            'scores': current_scores,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"[{datetime.now()}] Error getting scores: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'scores_count': len(current_scores)
    })

if __name__ == '__main__':
    # Generate initial scores
    current_scores = generate_semantic_scores("default")
    
    # Save initial scores to file
    with open('image_scores.json', 'w') as f:
        json.dump(current_scores, f, indent=2)
    
    print(f"[{datetime.now()}] Server starting...")
    print(f"[{datetime.now()}] Generated {len(current_scores)} initial scores")
    print(f"[{datetime.now()}] Server running on http://localhost:8000")
    
    app.run(debug=False, host='0.0.0.0', port=8000) 