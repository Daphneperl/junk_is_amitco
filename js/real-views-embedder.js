// Real Views Embedder - Embed Actual HTML Files as 3D Planes
class RealViewsEmbedder {
  constructor(scene, camera, renderer) {
    this.scene = scene;
    this.camera = camera;
    this.renderer = renderer;
    this.embeddedViews = new Map();
    this.viewPlanes = new Map();
    this.textureLoader = new THREE.TextureLoader();
    this.iframeContainer = null;

    // Create container for iframes
    this.createIframeContainer();
  }

  createIframeContainer() {
    // Create a container div for all iframes
    this.iframeContainer = document.createElement("div");
    this.iframeContainer.style.position = "absolute";
    this.iframeContainer.style.top = "0";
    this.iframeContainer.style.left = "0";
    this.iframeContainer.style.width = "100%";
    this.iframeContainer.style.height = "100%";
    this.iframeContainer.style.pointerEvents = "none";
    this.iframeContainer.style.zIndex = "1000";
    document.body.appendChild(this.iframeContainer);
  }

  // Embed all views from the views directory
  async embedAllViews() {
    console.log("Embedding all real views from views directory...");

    const viewConfigs = {
      artists: {
        name: "Artists",
        position: { x: 0, y: 0, z: 0 },
        scale: 1,
        rotation: { x: 0, y: 0, z: 0 },
        size: { width: 800, height: 600 },
        path: "views/Artists/Artist.html",
      },
      intimacy: {
        name: "Intimacy",
        position: { x: 500, y: 0, z: 0 },
        scale: 1,
        rotation: { x: 0, y: 0, z: 0 },
        size: { width: 800, height: 600 },
        path: "views/intimacy/Intimacy.html",
      },
      rhizome: {
        name: "Rhizome",
        position: { x: 0, y: 0, z: 500 },
        scale: 1,
        rotation: { x: 0, y: 0, z: 0 },
        size: { width: 800, height: 600 },
        path: "views/rhizome/rhizome.html",
      },
      temperament: {
        name: "Temperament",
        position: { x: 500, y: 0, z: 500 },
        scale: 1,
        rotation: { x: 0, y: 0, z: 0 },
        size: { width: 800, height: 600 },
        path: "views/temperament/temperament0606.html",
      },
      completeness: {
        name: "Completeness",
        position: { x: 0, y: 500, z: 0 },
        scale: 1,
        rotation: { x: 0, y: 0, z: 0 },
        size: { width: 800, height: 600 },
        path: "views/Completeness/Completeness.html",
      },
      hashtag: {
        name: "Hashtag Gallery",
        position: { x: 500, y: 500, z: 0 },
        scale: 1,
        rotation: { x: 0, y: 0, z: 0 },
        size: { width: 800, height: 600 },
        path: "views/hashtag_gallery/hashtag_gallery.html",
      },
      "open-question": {
        name: "Open Question",
        position: { x: 0, y: 500, z: 500 },
        scale: 1,
        rotation: { x: 0, y: 0, z: 0 },
        size: { width: 800, height: 600 },
        path: "views/open_question/openQuestion.html",
      },
      "total-galaxy": {
        name: "Total Galaxy",
        position: { x: 500, y: 500, z: 500 },
        scale: 1,
        rotation: { x: 0, y: 0, z: 0 },
        size: { width: 800, height: 600 },
        path: "views/Total_galaxy/total_galaxy_accurate.html",
      },
    };

    // Embed each view
    for (const [viewName, config] of Object.entries(viewConfigs)) {
      await this.embedView(viewName, config);
    }

    console.log("All real views embedded successfully!");
  }

  // Embed a single view
  async embedView(viewName, config) {
    try {
      console.log(`Embedding view: ${viewName} from ${config.path}`);

      // Create iframe
      const iframe = this.createIframe(config.path, config.size);

      // Create 3D plane to represent the view
      const plane = this.createViewPlane(config, iframe);

      // Store references
      this.embeddedViews.set(viewName, iframe);
      this.viewPlanes.set(viewName, plane);

      // Add plane to scene
      this.scene.add(plane);

      // Add view label
      const label = this.createViewLabel(config.name, config.position);
      this.scene.add(label);

      console.log(`Successfully embedded ${viewName}`);
    } catch (error) {
      console.error(`Error embedding view ${viewName}:`, error);
    }
  }

  // Create iframe for the view
  createIframe(path, size) {
    const iframe = document.createElement("iframe");
    iframe.src = path;
    iframe.style.width = `${size.width}px`;
    iframe.style.height = `${size.height}px`;
    iframe.style.border = "none";
    iframe.style.background = "transparent";
    iframe.style.pointerEvents = "auto";
    iframe.style.position = "absolute";
    iframe.style.top = "50%";
    iframe.style.left = "50%";
    iframe.style.transform = "translate(-50%, -50%)";
    iframe.style.zIndex = "1001";
    iframe.style.display = "none"; // Initially hidden

    // Add to container
    this.iframeContainer.appendChild(iframe);

    return iframe;
  }

  // Create 3D plane for the view
  createViewPlane(config, iframe) {
    const aspectRatio = config.size.width / config.size.height;
    const planeWidth = 400;
    const planeHeight = planeWidth / aspectRatio;

    // Create plane geometry
    const geometry = new THREE.PlaneGeometry(planeWidth, planeHeight);

    // Create material with placeholder texture
    const canvas = document.createElement("canvas");
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext("2d");

    // Create placeholder texture
    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#ffffff";
    ctx.font = "24px Arial";
    ctx.textAlign = "center";
    ctx.fillText(config.name, canvas.width / 2, canvas.height / 2);
    ctx.fillText("Loading...", canvas.width / 2, canvas.height / 2 + 30);

    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.MeshBasicMaterial({
      map: texture,
      transparent: true,
      opacity: 0.9,
      side: THREE.DoubleSide,
    });

    const plane = new THREE.Mesh(geometry, material);
    plane.position.set(config.position.x, config.position.y, config.position.z);
    plane.rotation.set(config.rotation.x, config.rotation.y, config.rotation.z);
    plane.scale.set(config.scale, config.scale, config.scale);

    // Store reference to iframe
    plane.userData = {
      viewName: config.name,
      iframe: iframe,
      config: config,
    };

    return plane;
  }

  // Create view label
  createViewLabel(name, position) {
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d");

    canvas.width = 256;
    canvas.height = 64;

    // Background
    context.fillStyle = "rgba(0, 0, 0, 0.8)";
    context.fillRect(0, 0, canvas.width, canvas.height);

    // Border
    context.strokeStyle = "rgba(255, 255, 255, 0.6)";
    context.lineWidth = 2;
    context.strokeRect(1, 1, canvas.width - 2, canvas.height - 2);

    // Text
    context.fillStyle = "white";
    context.font = "16px VT323";
    context.textAlign = "center";
    context.fillText(name, canvas.width / 2, canvas.height / 2 + 5);

    const texture = new THREE.CanvasTexture(canvas);
    const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
    const sprite = new THREE.Sprite(spriteMaterial);

    sprite.position.set(position.x, position.y + 250, position.z);
    sprite.scale.set(100, 25, 1);

    return sprite;
  }

  // Show iframe when plane is clicked or camera is close
  showIframe(viewName) {
    const iframe = this.embeddedViews.get(viewName);
    const plane = this.viewPlanes.get(viewName);

    if (iframe && plane) {
      // Hide all other iframes
      this.embeddedViews.forEach((otherIframe, otherName) => {
        if (otherName !== viewName) {
          otherIframe.style.display = "none";
        }
      });

      // Show this iframe
      iframe.style.display = "block";

      // Highlight the plane
      plane.material.opacity = 1.0;
      plane.material.emissive = new THREE.Color(0x333333);

      console.log(`Showing iframe for ${viewName}`);
    }
  }

  // Hide iframe
  hideIframe(viewName) {
    const iframe = this.embeddedViews.get(viewName);
    const plane = this.viewPlanes.get(viewName);

    if (iframe && plane) {
      iframe.style.display = "none";
      plane.material.opacity = 0.9;
      plane.material.emissive = new THREE.Color(0x000000);
    }
  }

  // Handle click on view plane
  handleClick(raycaster, mouse) {
    raycaster.setFromCamera(mouse, camera);

    // Check intersection with view planes
    const planes = Array.from(this.viewPlanes.values());
    const intersects = raycaster.intersectObjects(planes);

    if (intersects.length > 0) {
      const clickedPlane = intersects[0].object;
      const viewName = clickedPlane.userData.viewName;

      // Show the iframe for this view
      this.showIframe(viewName);

      return true; // Click was handled
    }

    return false; // Click was not handled
  }

  // Check camera distance and show/hide iframes accordingly
  updateIframeVisibility(cameraPosition) {
    this.viewPlanes.forEach((plane, viewName) => {
      const distance = cameraPosition.distanceTo(plane.position);
      const threshold = 300; // Show iframe when camera is within 300 units

      if (distance < threshold) {
        this.showIframe(viewName);
      } else {
        this.hideIframe(viewName);
      }
    });
  }

  // Clean up
  dispose() {
    // Remove all iframes
    this.embeddedViews.forEach((iframe) => {
      if (iframe.parentNode) {
        iframe.parentNode.removeChild(iframe);
      }
    });

    // Remove all planes from scene
    this.viewPlanes.forEach((plane) => {
      this.scene.remove(plane);
      plane.geometry.dispose();
      plane.material.dispose();
    });

    // Remove iframe container
    if (this.iframeContainer && this.iframeContainer.parentNode) {
      this.iframeContainer.parentNode.removeChild(this.iframeContainer);
    }

    this.embeddedViews.clear();
    this.viewPlanes.clear();
  }
}
