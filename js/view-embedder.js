// View Embedder System - Proper Integration with Existing Complex Views
class ViewEmbedder {
  constructor(scene, camera, renderer) {
    this.scene = scene;
    this.camera = camera;
    this.renderer = renderer;
    this.loadedViews = new Map();
    this.viewContainers = new Map();
    this.embeddedViews = new Map();
  }

  // Embed an existing view at specific coordinates
  async embedView(name, config) {
    console.log(`Embedding existing view: ${name}`);

    if (this.loadedViews.has(name)) {
      console.log(`View ${name} already embedded`);
      return;
    }

    try {
      // Create a container for this view
      const viewContainer = new THREE.Group();
      viewContainer.position.set(
        config.position.x,
        config.position.y,
        config.position.z
      );
      viewContainer.name = `embedded-view-${name}`;

      // Create a plane to display the view
      const planeGeometry = new THREE.PlaneGeometry(400, 300);
      const planeMaterial = new THREE.MeshBasicMaterial({
        color: 0x000000,
        transparent: true,
        opacity: 0.1,
      });
      const plane = new THREE.Mesh(planeGeometry, planeMaterial);
      plane.position.set(0, 0, 0);
      viewContainer.add(plane);

      // Add view label
      this.createViewLabel(config.name, viewContainer);

      // Add to scene
      this.scene.add(viewContainer);
      this.loadedViews.set(name, viewContainer);
      this.viewContainers.set(name, viewContainer);

      // Create the actual embedded view
      await this.createEmbeddedView(name, config, viewContainer);

      console.log(`Successfully embedded view: ${name}`);
    } catch (error) {
      console.error(`Error embedding view ${name}:`, error);
      this.createFallbackPlaceholder(name, config);
    }
  }

  // Create the actual embedded view
  async createEmbeddedView(name, config, container) {
    // Method 1: Create an iframe that loads the actual view
    const iframe = document.createElement("iframe");
    iframe.src = config.dataPath;
    iframe.style.width = "400px";
    iframe.style.height = "300px";
    iframe.style.border = "none";
    iframe.style.background = "transparent";
    iframe.style.position = "absolute";
    iframe.style.pointerEvents = "none"; // Prevent iframe interaction in 3D space

    // Create a canvas texture from the iframe (simplified approach)
    const canvas = document.createElement("canvas");
    canvas.width = 400;
    canvas.height = 300;
    const context = canvas.getContext("2d");

    // Create a placeholder texture for now
    context.fillStyle = "rgba(0, 0, 0, 0.8)";
    context.fillRect(0, 0, 400, 300);
    context.fillStyle = "white";
    context.font = "20px VT323";
    context.textAlign = "center";
    context.fillText(config.name, 200, 150);
    context.fillText("Click to open full view", 200, 180);

    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.MeshBasicMaterial({ map: texture });

    // Update the plane material
    const plane = container.children[0];
    plane.material = material;

    // Store the iframe reference for later use
    this.embeddedViews.set(name, {
      iframe: iframe,
      config: config,
      container: container,
    });

    // Add click handler to open the full view
    plane.userData = { viewName: name, config: config };
  }

  // Open the full view in a new window/tab
  openFullView(name) {
    const embeddedView = this.embeddedViews.get(name);
    if (embeddedView) {
      window.open(embeddedView.config.dataPath, "_blank");
    }
  }

  // Create a view label
  createViewLabel(text, container) {
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d");

    canvas.width = 256;
    canvas.height = 64;

    context.fillStyle = "rgba(0, 0, 0, 0.8)";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "white";
    context.font = "16px VT323";
    context.textAlign = "center";
    context.fillText(text, canvas.width / 2, canvas.height / 2 + 5);

    const texture = new THREE.CanvasTexture(canvas);
    const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
    const sprite = new THREE.Sprite(spriteMaterial);
    sprite.position.set(0, 200, 0);
    sprite.scale.set(100, 25, 1);
    container.add(sprite);
  }

  // Create fallback placeholder
  createFallbackPlaceholder(name, config) {
    const container = new THREE.Group();
    container.position.set(
      config.position.x,
      config.position.y,
      config.position.z
    );
    container.name = `embedded-view-${name}-fallback`;

    const geometry = new THREE.BoxGeometry(80, 80, 80);
    const material = new THREE.MeshBasicMaterial({
      color: 0x666666,
      wireframe: true,
      transparent: true,
      opacity: 0.4,
    });
    const placeholder = new THREE.Mesh(geometry, material);
    container.add(placeholder);

    this.createViewLabel(config.name, container);
    this.scene.add(container);
    this.loadedViews.set(name, container);
  }

  // Unload an embedded view
  async unloadView(name) {
    console.log(`Unloading embedded view: ${name}`);

    const viewContainer = this.loadedViews.get(name);
    if (viewContainer) {
      // Remove from scene
      this.scene.remove(viewContainer);

      // Dispose of geometries and materials
      viewContainer.traverse((child) => {
        if (child.geometry) {
          child.geometry.dispose();
        }
        if (child.material) {
          if (Array.isArray(child.material)) {
            child.material.forEach((material) => material.dispose());
          } else {
            child.material.dispose();
          }
        }
      });

      // Remove iframe if exists
      const embeddedView = this.embeddedViews.get(name);
      if (embeddedView && embeddedView.iframe) {
        embeddedView.iframe.remove();
      }

      // Remove from tracking
      this.loadedViews.delete(name);
      this.viewContainers.delete(name);
      this.embeddedViews.delete(name);

      console.log(`Successfully unloaded embedded view: ${name}`);
    }
  }

  // Handle click events on embedded views
  handleClick(raycaster, mouse) {
    raycaster.setFromCamera(mouse, this.camera);

    // Check all view containers
    const containers = Array.from(this.viewContainers.values());
    const intersects = raycaster.intersectObjects(containers, true);

    if (intersects.length > 0) {
      const clickedObject = intersects[0].object;
      if (clickedObject.userData && clickedObject.userData.viewName) {
        this.openFullView(clickedObject.userData.viewName);
        return true;
      }
    }

    return false;
  }
}
