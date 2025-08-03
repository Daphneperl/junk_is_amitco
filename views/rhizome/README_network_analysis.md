# Image Network Analysis

This directory contains scripts for analyzing semantic relationships between images and creating a network graph based on their artistic characteristics.

## Files

- `network_analysis.py` - Main script that creates network edges between images
- `network_stats.py` - Script to analyze and display network statistics
- `network_edges_images2.json` - Output file containing all network edges
- `network_env/` - Virtual environment with required dependencies

## How it works

The network analysis creates connections between images based on their semantic similarities:

### Node Representation

- Each image is a node in the network
- Images are identified by their filename (e.g., "image1", "image10", etc.)

### Edge Calculation

Edges are calculated based on two main factors:

1. **Keyword Similarity (80% weight)**

   - Exact keyword matches between images
   - Semantic similarity using TF-IDF vectorization
   - Confidence scores from the artistic analysis

2. **Vibe Similarity (20% weight)**
   - Exact vibe matches
   - Partial string matches
   - TF-IDF similarity for complex vibes

### Edge Weight Formula

```
Total Weight = (0.8 × Keyword Similarity) + (0.2 × Vibe Similarity)
```

## Network Statistics

The analysis of 665 images generated:

- **63,492 edges** with weight > 0.01
- **Average node degree**: 190.95
- **Weight range**: 0.0393 to 0.7581
- **Average weight**: 0.1828

## Usage

### Running the Analysis

```bash
# Activate virtual environment
source network_env/bin/activate

# Run the network analysis
python network_analysis.py
```

### Viewing Statistics

```bash
# Activate virtual environment
source network_env/bin/activate

# View network statistics
python network_stats.py
```

## Output Format

The `network_edges_images2.json` file contains:

```json
{
  "metadata": {
    "total_edges": 63492,
    "keyword_weight": 0.8,
    "vibe_weight": 0.2,
    "threshold": 0.01
  },
  "edges": [
    {
      "source": "image1",
      "target": "image101",
      "weight": 0.2,
      "keyword_similarity": 0.0,
      "vibe_similarity": 1.0
    }
  ]
}
```

## Dependencies

- `numpy` - Numerical computations
- `scikit-learn` - TF-IDF vectorization and similarity calculations
- `scipy` - Scientific computing utilities

## Input Data

The analysis uses the `artistic_analysis_images2.json` file which contains:

- Image filenames
- Keywords with confidence scores
- Vibe descriptions
- Image descriptions

## Strongest Connections

The top connections are typically between images that share:

- Multiple exact keyword matches
- High confidence keyword scores
- Similar semantic themes
- Matching vibes

This network can be used for:

- Visualizing image relationships
- Finding similar images
- Understanding semantic clusters
- Network analysis and community detection
