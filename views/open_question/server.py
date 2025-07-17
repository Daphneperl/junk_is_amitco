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

def generate_random_scores(query, image_files=None):
    """Generate random scores for images based on query"""
    random.seed(hash(query) % 2**32)  # Use query as seed for consistent results
    
    scores = {}
    
    # If no specific image files provided, generate for common formats
    if image_files is None:
        # Generate scores for the actual images used in the HTML (about 136 images)
        for i in range(5, 531):  # image5.jpg to image530.png
            if i <= 259:
                scores[f"image{i}.jpg"] = random.randint(1, 100)
            elif i <= 290:
                scores[f"image{i}.jpeg"] = random.randint(1, 100)
            elif i == 292 or i == 293:
                scores[f"image{i}.png"] = random.randint(1, 100)
            elif i >= 364 and i <= 530:
                scores[f"image{i}.png"] = random.randint(1, 100)
            elif i >= 499 and i <= 509:
                scores[f"image{i}.gif"] = random.randint(1, 100)
            elif i >= 510 and i <= 527:
                scores[f"image{i}.webp"] = random.randint(1, 100)
    else:
        # Generate scores only for the provided image files
        for filename in image_files:
            scores[filename] = random.randint(1, 100)
    
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
        current_scores = generate_random_scores(query)
        
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
                current_scores = generate_random_scores("default")
        
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
    current_scores = generate_random_scores("default")
    
    # Save initial scores to file
    with open('image_scores.json', 'w') as f:
        json.dump(current_scores, f, indent=2)
    
    print(f"[{datetime.now()}] Server starting...")
    print(f"[{datetime.now()}] Generated {len(current_scores)} initial scores")
    print(f"[{datetime.now()}] Server running on http://localhost:8000")
    
    app.run(debug=True, host='0.0.0.0', port=8000) 