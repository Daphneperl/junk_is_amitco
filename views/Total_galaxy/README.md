# Total Galaxy - Unified Visualization

This is a unified 3D galaxy view that combines all the different visualizations from the project into a single immersive 3D space.

## Features

### Galaxy Layout

- **Rhizome**: Keyword-based clustering at the center (0, 0, 0)
- **Artist**: Helix structure at (800, 200, 400)
- **Open Question**: Layered score circles at (-600, 300, 500)
- **Hashtag Gallery**: Curved wall at (400, -400, 600)
- **Completeness**: Score-based arrangement at (-400, -600, 300)
- **Intimacy**: Tunnel structure at (600, 500, -400)
- **Temperature**: Score-based spiral at (-800, 100, -200)

### Navigation

- **Mouse/Trackpad**: Orbit around the galaxy
- **Scroll**: Zoom in/out
- **View Labels**: Click on the left panel to navigate to specific views
- **Camera Info**: Real-time position display in bottom-right corner

### Controls

- **Search Bar**: Type to search (placeholder functionality)
- **Upload Button**: Access upload functionality (placeholder)
- **Smooth Transitions**: Automatic camera movement between views

### Visual Effects

- **Particle System**: 2000 animated particles throughout the galaxy
- **Vignette Effect**: Subtle darkening around edges
- **Noise Background**: Dynamic texture background
- **Gentle Animations**: Each view rotates slowly for dynamic effect

## Technical Details

### Image Loading

- Preloads all images from `../../images/images.json`
- Uses artistic analysis data from `../../image_analysis/artistic_analysis_filtered.json`
- Handles missing images gracefully with placeholders

### Performance

- Optimized rendering with Three.js
- Efficient particle system
- Memory management with proper cleanup
- WebGL context loss handling

### File Structure

```
views/Total_galaxy/
├── total_galaxy.html    # Main visualization file
└── README.md           # This file
```

## Usage

1. Open `total_galaxy.html` in a web browser
2. Wait for the loading screen to complete
3. Use mouse/trackpad to navigate the galaxy
4. Click view labels to jump to specific visualizations
5. Explore the different arrangements and patterns

## Dependencies

- Three.js r128
- D3.js v7
- Custom fonts (Pixel, VT323, Heming)
- Image assets from the main project

## Browser Compatibility

- Modern browsers with WebGL support
- Chrome, Firefox, Safari, Edge recommended
- Mobile browsers may have performance limitations
