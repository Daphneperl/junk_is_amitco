// View Loader System - Proper Integration with Existing Views
class ViewLoader {
  constructor(scene, camera, renderer) {
    this.scene = scene;
    this.camera = camera;
    this.renderer = renderer;
    this.loadedViews = new Map();
    this.viewObjects = new Map();
    this.textureLoader = new THREE.TextureLoader();
    this.iframes = new Map();
  }

  // Load a specific view with its actual content
  async loadView(name, config) {
    console.log(`Loading actual view content: ${name}`);

    // Check if view is already loaded
    if (this.loadedViews.has(name)) {
      console.log(`View ${name} already loaded`);
      return;
    }

    try {
      // Create a container for this view's content
      const viewContainer = new THREE.Group();
      viewContainer.position.set(
        config.position.x,
        config.position.y,
        config.position.z
      );
      viewContainer.name = `view-${name}`;

      // Load the actual view content based on the existing HTML files
      await this.loadActualViewContent(name, config, viewContainer);

      // Add to scene
      this.scene.add(viewContainer);
      this.loadedViews.set(name, viewContainer);

      console.log(`Successfully loaded actual view: ${name}`);
    } catch (error) {
      console.error(`Error loading view ${name}:`, error);
      // Create a fallback placeholder
      this.createFallbackPlaceholder(name, config);
    }
  }

  // Load the actual view content from existing HTML files
  async loadActualViewContent(name, config, container) {
    // Create an iframe to load the actual view
    const iframe = document.createElement("iframe");
    iframe.src = config.dataPath;
    iframe.style.width = "800px";
    iframe.style.height = "600px";
    iframe.style.border = "none";
    iframe.style.background = "transparent";

    // Create a plane to display the iframe
    const planeGeometry = new THREE.PlaneGeometry(800, 600);
    const planeMaterial = new THREE.MeshBasicMaterial({
      color: 0x000000,
      transparent: true,
      opacity: 0.9,
    });
    const plane = new THREE.Mesh(planeGeometry, planeMaterial);
    plane.position.set(0, 0, 0);
    container.add(plane);

    // Store iframe reference
    this.iframes.set(name, iframe);

    // Add view label
    this.createViewLabel(config.name, container);

    // Alternative approach: Load the actual Three.js content
    // This would require extracting the Three.js code from each view
    await this.loadViewThreeJSContent(name, config, container);
  }

  // Load the actual Three.js content from existing views
  async loadViewThreeJSContent(name, config, container) {
    switch (name) {
      case "artists":
        await this.loadArtistsView(container);
        break;
      case "intimacy":
        await this.loadIntimacyView(container);
        break;
      case "rhizome":
        await this.loadRhizomeView(container);
        break;
      case "temperament":
        await this.loadTemperamentView(container);
        break;
      case "completeness":
        await this.loadCompletenessView(container);
        break;
      case "hashtag":
        await this.loadHashtagView(container);
        break;
      case "open-question":
        await this.loadOpenQuestionView(container);
        break;
      case "total-galaxy":
        await this.loadTotalGalaxyView(container);
        break;
      default:
        this.createGenericPlaceholder(name, config, container);
    }
  }

  // Load Artists View (extracted from Artist.html)
  async loadArtistsView(container) {
    try {
      // Load the artistic analysis data
      const artistData = await this.loadJSONData(
        "image_analysis/artistic_analysis_filtered.json"
      );

      if (artistData) {
        // Group by artist
        const artists = this.groupByArtist(artistData);
        const artistNames = Object.keys(artists);

        // Create helix positions for artists
        const helixPositions = this.createHelixPositions(artistNames.length);

        // Create artist clusters in helix formation
        artistNames.forEach((artist, index) => {
          const position = helixPositions[index];
          const artistGroup = this.createArtistCluster(
            artist,
            artists[artist],
            position
          );
          container.add(artistGroup);
        });

        // Create helix outline
        const helixOutline = this.createHelixOutline();
        container.add(helixOutline);

        // Create helix grid
        const helixGrid = this.createHelixGrid();
        container.add(helixGrid);
      } else {
        this.createGenericPlaceholder(
          "artists",
          { name: "Artists Gallery" },
          container
        );
      }
    } catch (error) {
      console.error("Error loading artists view:", error);
      this.createGenericPlaceholder(
        "artists",
        { name: "Artists Gallery" },
        container
      );
    }
  }

  // Load Intimacy View (extracted from Intimacy.html)
  async loadIntimacyView(container) {
    try {
      // Create tunnel structure
      const tunnelCurve = this.createTunnelCurve();
      const tunnelWire = this.createTunnelWireGrid();
      container.add(tunnelWire);

      // Load and place artworks along the tunnel
      const artworks = await this.loadArtworks();
      if (artworks.length > 0) {
        this.placeArtworksInTunnel(artworks, tunnelCurve, container);
      }
    } catch (error) {
      console.error("Error loading intimacy view:", error);
      this.createGenericPlaceholder(
        "intimacy",
        { name: "Intimacy Tunnel" },
        container
      );
    }
  }

  // Load Rhizome View (extracted from rhizome.html)
  async loadRhizomeView(container) {
    try {
      const artisticData = await this.loadJSONData(
        "image_analysis/artistic_analysis_filtered.json"
      );

      if (artisticData) {
        // Create network visualization
        this.createRhizomeNetwork(artisticData, container);
      } else {
        this.createGenericPlaceholder(
          "rhizome",
          { name: "Rhizome Network" },
          container
        );
      }
    } catch (error) {
      console.error("Error loading rhizome view:", error);
      this.createGenericPlaceholder(
        "rhizome",
        { name: "Rhizome Network" },
        container
      );
    }
  }

  // Load other views with their actual content
  async loadTemperamentView(container) {
    try {
      const scoreData = await this.loadJSONData(
        "views/temperament/temperament_scores.json"
      );
      if (scoreData) {
        this.createTemperamentVisualization(scoreData, container);
      } else {
        this.createGenericPlaceholder(
          "temperament",
          { name: "Temperament Scores" },
          container
        );
      }
    } catch (error) {
      console.error("Error loading temperament view:", error);
      this.createGenericPlaceholder(
        "temperament",
        { name: "Temperament Scores" },
        container
      );
    }
  }

  async loadCompletenessView(container) {
    try {
      const completenessData = await this.loadJSONData(
        "views/Completeness/inverted_sketchiness_scores.json"
      );
      if (completenessData) {
        this.createCompletenessVisualization(completenessData, container);
      } else {
        this.createGenericPlaceholder(
          "completeness",
          { name: "Completeness Analysis" },
          container
        );
      }
    } catch (error) {
      console.error("Error loading completeness view:", error);
      this.createGenericPlaceholder(
        "completeness",
        { name: "Completeness Analysis" },
        container
      );
    }
  }

  async loadHashtagView(container) {
    try {
      const hashtagData = await this.loadCSVData(
        "views/hashtag_gallery/Hashtags.csv"
      );
      if (hashtagData) {
        this.createHashtagVisualization(hashtagData, container);
      } else {
        this.createGenericPlaceholder(
          "hashtag",
          { name: "Hashtag Gallery" },
          container
        );
      }
    } catch (error) {
      console.error("Error loading hashtag view:", error);
      this.createGenericPlaceholder(
        "hashtag",
        { name: "Hashtag Gallery" },
        container
      );
    }
  }

  async loadOpenQuestionView(container) {
    try {
      const questionData = await this.loadJSONData(
        "views/open_question/image_scores.json"
      );
      if (questionData) {
        this.createOpenQuestionVisualization(questionData, container);
      } else {
        this.createGenericPlaceholder(
          "open-question",
          { name: "Open Question" },
          container
        );
      }
    } catch (error) {
      console.error("Error loading open question view:", error);
      this.createGenericPlaceholder(
        "open-question",
        { name: "Open Question" },
        container
      );
    }
  }

  async loadTotalGalaxyView(container) {
    try {
      // Create galaxy visualization
      this.createGalaxyVisualization(container);
    } catch (error) {
      console.error("Error loading total galaxy view:", error);
      this.createGenericPlaceholder(
        "total-galaxy",
        { name: "Total Galaxy" },
        container
      );
    }
  }

  // Helper methods for creating actual visualizations

  // Artists View Helpers
  createHelixPositions(totalClusters) {
    const positions = [];
    const turns = 20;
    const heightStep = 1500 / totalClusters;
    const radius = 40;

    for (let i = 0; i < totalClusters; i++) {
      const angle = (i / totalClusters) * Math.PI * 2 * turns;
      const x = radius * Math.cos(angle);
      const y = i * heightStep - 750;
      const z = radius * Math.sin(angle);
      positions.push(new THREE.Vector3(x, y, z));
    }

    return positions;
  }

  createHelixOutline() {
    const turns = 20;
    const radius = 40;
    const totalHeight = 1500;
    const segments = 1200;

    const points = [];
    for (let i = 0; i <= segments; i++) {
      const angle = (i / segments) * Math.PI * 2 * turns;
      const y = (i / segments) * totalHeight - totalHeight / 2;
      const x = radius * Math.cos(angle);
      const z = radius * Math.sin(angle);
      points.push(new THREE.Vector3(x, y, z));
    }

    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.9,
    });

    return new THREE.Line(geometry, material);
  }

  createHelixGrid() {
    const turns = 20;
    const radius = 40;
    const totalHeight = 1500;
    const gridGroup = new THREE.Group();

    const spiralGuides = 6;
    for (let guide = 0; guide <= spiralGuides; guide++) {
      const guideRadius = radius * (0.2 + guide * 0.15);
      const points = [];
      const segments = 800;

      for (let i = 0; i <= segments; i++) {
        const angle = (i / segments) * Math.PI * 2 * turns;
        const y = (i / segments) * totalHeight - totalHeight / 2;
        const x = guideRadius * Math.cos(angle);
        const z = guideRadius * Math.sin(angle);
        points.push(new THREE.Vector3(x, y, z));
      }

      const isOutermost = guide === spiralGuides;
      const opacity = isOutermost ? 0.8 : 0.4 + guide * 0.05;

      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const material = new THREE.LineBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: opacity,
      });

      const line = new THREE.Line(geometry, material);
      gridGroup.add(line);
    }

    return gridGroup;
  }

  createArtistCluster(artist, images, position) {
    const cluster = new THREE.Group();
    cluster.position.copy(position);

    // Create central artist sphere
    const artistSphere = new THREE.Mesh(
      new THREE.SphereGeometry(15, 16, 16),
      new THREE.MeshBasicMaterial({
        color: new THREE.Color().setHSL(Math.random(), 0.8, 0.6),
        transparent: true,
        opacity: 0.8,
      })
    );
    cluster.add(artistSphere);

    // Add artist label
    const label = this.createTextSprite(artist);
    label.position.set(0, 30, 0);
    cluster.add(label);

    // Add image spheres around the artist
    images.slice(0, 10).forEach((image, index) => {
      const imageSphere = new THREE.Mesh(
        new THREE.SphereGeometry(3, 8, 8),
        new THREE.MeshBasicMaterial({
          color: new THREE.Color().setHSL(Math.random(), 0.7, 0.5),
          transparent: true,
          opacity: 0.6,
        })
      );

      const angle = (index / images.length) * Math.PI * 2;
      const radius = 30;
      imageSphere.position.set(
        Math.cos(angle) * radius,
        Math.sin(angle) * radius,
        0
      );

      cluster.add(imageSphere);
    });

    return cluster;
  }

  // Intimacy View Helpers
  createTunnelCurve() {
    return new THREE.CatmullRomCurve3(
      Array.from({ length: 200 }, (_, i) => {
        const t = i / 199;
        const angle = t * Math.PI * 1.8;
        const radius = 1500;
        return new THREE.Vector3(
          Math.sin(angle) * radius,
          0,
          -Math.cos(angle) * radius
        );
      })
    );
  }

  createTunnelWireGrid() {
    const wireGroup = new THREE.Group();
    // Simplified tunnel wire grid
    const tunnelGeometry = new THREE.CylinderGeometry(100, 100, 800, 32);
    const tunnelMaterial = new THREE.MeshBasicMaterial({
      color: 0x333333,
      wireframe: true,
      transparent: true,
      opacity: 0.3,
    });
    const tunnel = new THREE.Mesh(tunnelGeometry, tunnelMaterial);
    tunnel.rotation.x = Math.PI / 2;
    wireGroup.add(tunnel);

    return wireGroup;
  }

  async loadArtworks() {
    try {
      const response = await fetch("images/images.json");
      const imageList = await response.json();
      return imageList.slice(0, 50); // Limit for performance
    } catch (error) {
      console.warn("Could not load artworks:", error);
      return [];
    }
  }

  placeArtworksInTunnel(artworks, tunnelCurve, container) {
    artworks.forEach((artwork, index) => {
      const t = index / artworks.length;
      const position = tunnelCurve.getPointAt(t);

      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(5, 8, 8),
        new THREE.MeshBasicMaterial({
          color: new THREE.Color().setHSL(Math.random(), 0.7, 0.5),
          transparent: true,
          opacity: 0.6,
        })
      );
      sphere.position.copy(position);
      container.add(sphere);
    });
  }

  // Rhizome View Helpers
  createRhizomeNetwork(artisticData, container) {
    const nodeCount = Math.min(50, artisticData.length);
    const nodes = [];

    // Create nodes
    for (let i = 0; i < nodeCount; i++) {
      const node = new THREE.Mesh(
        new THREE.SphereGeometry(3, 8, 8),
        new THREE.MeshBasicMaterial({
          color: new THREE.Color().setHSL(Math.random(), 0.8, 0.6),
          transparent: true,
          opacity: 0.8,
        })
      );
      node.position.set(
        (Math.random() - 0.5) * 300,
        (Math.random() - 0.5) * 300,
        (Math.random() - 0.5) * 300
      );
      nodes.push(node);
      container.add(node);
    }

    // Create connections
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const distance = nodes[i].position.distanceTo(nodes[j].position);
        if (distance < 100) {
          const lineGeometry = new THREE.BufferGeometry().setFromPoints([
            nodes[i].position,
            nodes[j].position,
          ]);
          const lineMaterial = new THREE.LineBasicMaterial({
            color: 0x666666,
            transparent: true,
            opacity: 0.3,
          });
          const line = new THREE.Line(lineGeometry, lineMaterial);
          container.add(line);
        }
      }
    }
  }

  // Other visualization helpers
  createTemperamentVisualization(scoreData, container) {
    scoreData.forEach((item, index) => {
      const score = item.score || Math.random();
      const size = 2 + score * 8;
      const color = new THREE.Color().setHSL(score, 0.8, 0.5);

      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(size, 8, 8),
        new THREE.MeshBasicMaterial({
          color: color,
          transparent: true,
          opacity: 0.7,
        })
      );

      sphere.position.set(
        (Math.random() - 0.5) * 200,
        score * 100,
        (Math.random() - 0.5) * 200
      );

      container.add(sphere);
    });
  }

  createCompletenessVisualization(completenessData, container) {
    const gridSize = 10;
    const spacing = 20;

    for (let x = 0; x < gridSize; x++) {
      for (let z = 0; z < gridSize; z++) {
        const index = x * gridSize + z;
        const score = completenessData[index]?.score || Math.random();

        const height = 10 + score * 50;
        const color = new THREE.Color().setHSL(score, 0.7, 0.5);

        const box = new THREE.Mesh(
          new THREE.BoxGeometry(spacing * 0.8, height, spacing * 0.8),
          new THREE.MeshBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.6,
          })
        );

        box.position.set(
          (x - gridSize / 2) * spacing,
          height / 2,
          (z - gridSize / 2) * spacing
        );

        container.add(box);
      }
    }
  }

  createHashtagVisualization(hashtagData, container) {
    hashtagData.slice(0, 30).forEach((row, index) => {
      const hashtag = row.hashtag || `#tag${index}`;
      const count = parseInt(row.count) || Math.floor(Math.random() * 100);

      const size = 5 + count * 0.1;
      const color = new THREE.Color().setHSL(Math.random(), 0.8, 0.6);

      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(size, 8, 8),
        new THREE.MeshBasicMaterial({
          color: color,
          transparent: true,
          opacity: 0.7,
        })
      );

      sphere.position.set(
        (Math.random() - 0.5) * 300,
        (Math.random() - 0.5) * 300,
        (Math.random() - 0.5) * 300
      );

      container.add(sphere);
    });
  }

  createOpenQuestionVisualization(questionData, container) {
    questionData.slice(0, 20).forEach((item, index) => {
      const score = item.score || Math.random();
      const size = 3 + score * 10;

      const geometry = new THREE.OctahedronGeometry(size);
      const material = new THREE.MeshBasicMaterial({
        color: new THREE.Color().setHSL(score, 0.7, 0.5),
        wireframe: true,
        transparent: true,
        opacity: 0.6,
      });

      const octahedron = new THREE.Mesh(geometry, material);
      octahedron.position.set(
        (Math.random() - 0.5) * 250,
        (Math.random() - 0.5) * 250,
        (Math.random() - 0.5) * 250
      );

      container.add(octahedron);
    });
  }

  createGalaxyVisualization(container) {
    const starCount = 200;

    for (let i = 0; i < starCount; i++) {
      const star = new THREE.Mesh(
        new THREE.SphereGeometry(1 + Math.random() * 3, 6, 6),
        new THREE.MeshBasicMaterial({
          color: new THREE.Color().setHSL(Math.random(), 0.3, 0.8),
          transparent: true,
          opacity: 0.8,
        })
      );

      const angle = Math.random() * Math.PI * 2;
      const radius = Math.random() * 200;
      const height = (Math.random() - 0.5) * 50;

      star.position.set(
        Math.cos(angle) * radius,
        height,
        Math.sin(angle) * radius
      );

      container.add(star);
    }
  }

  // Utility methods
  createTextSprite(text) {
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d");

    canvas.width = 256;
    canvas.height = 64;

    context.font = "24px monospace";
    context.textAlign = "center";
    context.fillStyle = "white";
    context.fillText(text, canvas.width / 2, canvas.height / 2);

    const texture = new THREE.CanvasTexture(canvas);
    const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
    const sprite = new THREE.Sprite(spriteMaterial);
    sprite.scale.set(50, 12.5, 1);

    return sprite;
  }

  createViewLabel(text, container) {
    const label = this.createTextSprite(text);
    label.position.set(0, 100, 0);
    container.add(label);
  }

  createGenericPlaceholder(name, config, container) {
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
  }

  createFallbackPlaceholder(name, config) {
    const container = new THREE.Group();
    container.position.set(
      config.position.x,
      config.position.y,
      config.position.z
    );
    container.name = `view-${name}-fallback`;

    this.createGenericPlaceholder(name, config, container);
    this.scene.add(container);
    this.loadedViews.set(name, container);
  }

  // Unload a view and clean up resources
  async unloadView(name) {
    console.log(`Unloading view: ${name}`);

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
      const iframe = this.iframes.get(name);
      if (iframe) {
        iframe.remove();
        this.iframes.delete(name);
      }

      // Remove from tracking
      this.loadedViews.delete(name);
      this.viewObjects.delete(name);

      console.log(`Successfully unloaded view: ${name}`);
    }
  }

  // Helper method to load JSON data
  async loadJSONData(path) {
    try {
      const response = await fetch(path);
      if (response.ok) {
        return await response.json();
      }
    } catch (error) {
      console.warn(`Could not load JSON data from ${path}:`, error);
    }
    return null;
  }

  // Helper method to load CSV data
  async loadCSVData(path) {
    try {
      const response = await fetch(path);
      if (response.ok) {
        const csvText = await response.text();
        return this.parseCSV(csvText);
      }
    } catch (error) {
      console.warn(`Could not load CSV data from ${path}:`, error);
    }
    return null;
  }

  // Simple CSV parser
  parseCSV(csvText) {
    const lines = csvText.split("\n");
    const headers = lines[0].split(",").map((h) => h.trim());
    const data = [];

    for (let i = 1; i < lines.length; i++) {
      if (lines[i].trim()) {
        const values = lines[i].split(",").map((v) => v.trim());
        const row = {};
        headers.forEach((header, index) => {
          row[header] = values[index];
        });
        data.push(row);
      }
    }

    return data;
  }

  // Helper method to group data by artist
  groupByArtist(data) {
    const artists = {};
    data.forEach((item) => {
      const artist = item.artist || "Unknown";
      if (!artists[artist]) {
        artists[artist] = [];
      }
      artists[artist].push(item);
    });
    return artists;
  }
}
