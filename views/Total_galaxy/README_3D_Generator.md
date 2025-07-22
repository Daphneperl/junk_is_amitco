# 3D View Generator

A 3D space navigation system that allows you to explore all your visualization views in a single 3D environment.

## Features

- **3D Grid Layout**: All views are positioned in a 3D grid with 500-unit spacing
- **Smooth Camera Transitions**: 2-second smooth camera movements between views
- **View Unloading**: Memory-efficient system that unloads distant views
- **Interactive Navigation**: Click on view markers or use the navigation panel
- **Real-time Info**: Camera position and current view information display

## How to Use

1. **Open the Generator**: Open `3d_view_generator.html` in your browser
2. **Navigate Between Views**:
   - Click on the spherical markers in 3D space
   - Use the navigation panel on the left side
   - Use mouse/trackpad to orbit around the 3D space
3. **View Information**:
   - Current view and position info is shown at bottom-left
   - Camera position and distance info is shown at bottom-right

## View Layout

The views are arranged in a 3D grid as follows:

```
Y=500: [completeness] [hashtag]     [open-question] [total-galaxy]
Y=0:   [artists]     [intimacy]     [rhizome]       [temperament]
```

Each view is positioned 500 units apart in X, Y, and Z coordinates.

## View Types

- **Artists Gallery**: Artist clusters with image representations
- **Intimacy Tunnel**: Cylindrical tunnel with floating intimacy elements
- **Rhizome Network**: Network nodes with connecting lines
- **Temperament Scores**: 3D scatter plot of temperament data
- **Completeness Analysis**: 3D grid of completeness scores
- **Hashtag Gallery**: Floating hashtag elements
- **Open Question**: Octahedron elements representing questions
- **Total Galaxy**: Star field galaxy visualization

## Technical Details

### Architecture

- **View3DGenerator**: Main controller for view management and transitions
- **ViewLoader**: Handles loading/unloading of view-specific content
- **Three.js**: 3D rendering and scene management
- **Raycasting**: Mouse interaction with 3D objects

### Memory Management

- Views are loaded on-demand when you navigate to them
- Distant views are automatically unloaded to save memory
- Geometries and materials are properly disposed when unloading

### Performance

- Uses efficient Three.js rendering
- Implements proper resource cleanup
- Smooth 60fps animations with easing functions

## Customization

### Adding New Views

1. Add a new entry to `viewConfigs` in the main HTML file
2. Implement the corresponding loading method in `ViewLoader`
3. Position the view in the 3D grid

### Modifying View Positions

Edit the `position` property in `viewConfigs`:

```javascript
"new-view": {
    position: { x: 1000, y: 0, z: 0 }, // 1000 units in X direction
    name: "New View",
    type: "new-gallery-type",
    // ... other properties
}
```

### Changing Transition Speed

Modify the `transitionDuration` property in the `View3DGenerator` class:

```javascript
this.transitionDuration = 3000; // 3 seconds instead of 2
```

## Browser Compatibility

- Modern browsers with WebGL support
- Chrome, Firefox, Safari, Edge (latest versions)
- Requires JavaScript enabled

## File Structure

```
├── 3d_view_generator.html    # Main application
├── README_3D_Generator.md    # This documentation
├── total_galaxy_accurate.html # Total galaxy view
├── total_galaxy_simple.html  # Simple galaxy view
├── total_galaxy.html         # Original galaxy view
├── README.md                 # Galaxy documentation
├── ../js/                    # JavaScript utilities (parent directory)
├── ../views/                 # Individual view folders (parent directory)
├── ../assets/                # Fonts and images (parent directory)
└── ../image_analysis/        # Data files (parent directory)
```

## Troubleshooting

### Views Not Loading

- Check browser console for errors
- Ensure data files exist in the correct paths
- Verify CORS settings if running from a server

### Performance Issues

- Reduce the number of objects in view content
- Increase view unloading distance
- Check for memory leaks in browser dev tools

### Navigation Problems

- Ensure Three.js and dependencies are loaded
- Check that raycaster is working properly
- Verify view markers have correct userData

## Future Enhancements

- [ ] Keyboard shortcuts for navigation
- [ ] View previews/thumbnails
- [ ] Mini-map showing all view locations
- [ ] Custom view layouts and arrangements
- [ ] View state preservation
- [ ] Advanced lighting and effects
