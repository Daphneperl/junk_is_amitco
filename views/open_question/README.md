# Open Question Interactive Search System

This system provides an interactive search interface for the 3D image visualization. Users can type queries in the search bar, and the Python server will generate new scores for the images, causing the visualization to reorganize in real-time.

## Files

- `server.py` - Flask server that handles search queries and generates scores
- `openQuestion_with_search.html` - Modified HTML file with search functionality
- `requirements.txt` - Python dependencies
- `start_server.py` - Easy startup script
- `image_scores.json` - Generated file containing image scores (created automatically)

## Quick Start

1. **Start the server:**
   ```bash
   cd views/open_question
   python start_server.py
   ```

2. **Open the visualization:**
   - Open `openQuestion_with_search.html` in your web browser
   - The server will be running on `http://localhost:8000`

3. **Use the search:**
   - Type any query in the search bar at the bottom of the page
   - The visualization will reorganize based on new random scores
   - You'll see status messages indicating the query was processed

## How It Works

1. **Search Input:** Users type queries in the search bar
2. **Server Processing:** The Flask server receives the query and generates new random scores for all images
3. **Real-time Update:** The HTML page receives the new scores and reorganizes the 3D visualization
4. **Visual Feedback:** Status messages and loading indicators show the process

## API Endpoints

- `POST /search` - Submit a search query
  - Body: `{"query": "your search term"}`
  - Returns: Updated scores for all images

- `GET /scores` - Get current scores
  - Returns: Current image scores

- `GET /health` - Health check
  - Returns: Server status

## Features

- **Debounced Search:** Waits 500ms after user stops typing before searching
- **Real-time Updates:** Visualization reorganizes immediately after receiving new scores
- **Status Feedback:** Shows search status and loading indicators
- **Error Handling:** Graceful error handling with user feedback
- **Consistent Results:** Same query always produces the same scores (using hash-based seeding)

## Customization

To modify the scoring algorithm:
1. Edit the `generate_random_scores()` function in `server.py`
2. Replace the random scoring with your actual image analysis logic
3. The system will automatically use your new scoring method

## Troubleshooting

- **Server won't start:** Make sure you have Python 3.6+ installed
- **CORS errors:** The server includes CORS headers, but if you have issues, check your browser's developer console
- **Images not loading:** Ensure the relative path to the images folder is correct (`../../images/`)

## Development

The system is designed to be easily extensible:
- Add new analysis methods to the server
- Modify the visualization layout in the HTML
- Add new search features or filters 