# Artist View - Creator-Based Clustering

This view has been updated to use creator-based clustering instead of random image grouping.

## Files

- `Artist.html` - The main visualization file
- `create_creator_clusters.py` - Script to generate creator clusters from DF.csv
- `creator_clusters.json` - Generated creator clusters data
- `README.md` - This documentation file

## How it works

1. **Data Source**: The system reads from `assets/DF.csv` which contains image metadata including creator information
2. **Image Source**: Images are loaded from the `images2/` folder
3. **Filtering**: Entries where the creator is "-" are ignored
4. **Clustering**: Images are grouped by their creator (from the `@creator` column)
5. **Visualization**: Each creator's images are displayed as a cluster in the 3D helix

## Setup

1. Ensure `assets/DF.csv` exists with the correct structure
2. Ensure `images2/` folder contains the image files
3. Run the clustering script:
   ```bash
   python3 create_creator_clusters.py
   ```
4. Open `Artist.html` in a web browser

## Features

- **Creator Names**: Each cluster shows the actual creator name (e.g., "@pink_flamingo")
- **Image Details**: Click on images in the cluster popup to see detailed metadata
- **Rich Data**: Displays title, location, date, context, keywords, and other metadata
- **3D Navigation**: Scroll and rotate through the helix of creator clusters

## Data Structure

The `creator_clusters.json` file contains:

```json
[
  {
    "creator": "@username",
    "images": [
      {
        "filename": "image1.jpg",
        "title": "Image Title",
        "location": "Location",
        "exact_spot": "Exact Spot",
        "intimacy_level": "Level",
        "date_stamp": "Date",
        "daytime_icon": "Icon",
        "rawness_percent": "Percentage",
        "context": "Context",
        "artistic_description": "Description",
        "keywords": "keyword1; keyword2; keyword3",
        "keyword_confidences": "0.8; 0.7; 0.6",
        "vibe": "vibe"
      }
    ]
  }
]
```

## Troubleshooting

- If you get an error about missing `creator_clusters.json`, run the Python script first
- If images don't load, check that the image files exist in the `images2/` folder
- If the CSV structure changes, update the Python script accordingly
