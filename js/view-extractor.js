// View Extractor System - Extract and Integrate Existing Three.js Content
class ViewExtractor {
  constructor(scene, camera, renderer) {
    this.scene = scene;
    this.camera = camera;
    this.renderer = renderer;
    this.loadedViews = new Map();
    this.viewContainers = new Map();
    this.extractedContent = new Map();
  }

  // Extract and embed an existing view at specific coordinates
  async extractAndEmbedView(name, config) {
    console.log(`Extracting and embedding view: ${name}`);

    if (this.loadedViews.has(name)) {
      console.log(`View ${name} already extracted`);
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
      viewContainer.name = `extracted-view-${name}`;

      // Extract the actual Three.js content based on view type
      await this.extractViewContent(name, config, viewContainer);

      // Add to scene
      this.scene.add(viewContainer);
      this.loadedViews.set(name, viewContainer);
      this.viewContainers.set(name, viewContainer);

      console.log(`Successfully extracted and embedded view: ${name}`);
    } catch (error) {
      console.error(`Error extracting view ${name}:`, error);
      this.createFallbackPlaceholder(name, config);
    }
  }

  // Extract the actual Three.js content from existing views
  async extractViewContent(name, config, container) {
    switch (name) {
      case "artists":
        await this.extractArtistsView(container);
        break;
      case "intimacy":
        await this.extractIntimacyView(container);
        break;
      case "rhizome":
        await this.extractRhizomeView(container);
        break;
      case "temperament":
        await this.extractTemperamentView(container);
        break;
      case "completeness":
        await this.extractCompletenessView(container);
        break;
      case "hashtag":
        await this.extractHashtagView(container);
        break;
      case "open-question":
        await this.extractOpenQuestionView(container);
        break;
      case "total-galaxy":
        await this.extractTotalGalaxyView(container);
        break;
      default:
        this.createGenericPlaceholder(name, config, container);
    }
  }

  // Extract Artists View (from Artist.html)
  async extractArtistsView(container) {
    try {
      // Load the artistic analysis data
      const artistData = await this.loadJSONData(
        "image_analysis/artistic_analysis_filtered.json"
      );

      if (artistData) {
        // Group by artist
        const artists = this.groupByArtist(artistData);
        const artistNames = Object.keys(artists);

        // Create helix positions for artists (from Artist.html)
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

        // Create helix outline (from Artist.html)
        const helixOutline = this.createHelixOutline();
        container.add(helixOutline);

        // Create helix grid (from Artist.html)
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
      console.error("Error extracting artists view:", error);
      this.createGenericPlaceholder(
        "artists",
        { name: "Artists Gallery" },
        container
      );
    }
  }

  // Extract Intimacy View (from Intimacy.html)
  async extractIntimacyView(container) {
    try {
      // Create tunnel curve (from Intimacy.html)
      const tunnelCurve = this.createTunnelCurve();

      // Create tunnel wire grid (from Intimacy.html)
      const tunnelWire = this.createTunnelWireGrid();
      container.add(tunnelWire);

      // Load and place artworks along the tunnel (from Intimacy.html)
      const artworks = await this.loadArtworks();
      if (artworks.length > 0) {
        this.placeArtworksInTunnel(artworks, tunnelCurve, container);
      }

      // Add station markers (from Intimacy.html)
      this.addStationMarkers(container);
    } catch (error) {
      console.error("Error extracting intimacy view:", error);
      this.createGenericPlaceholder(
        "intimacy",
        { name: "Intimacy Tunnel" },
        container
      );
    }
  }

  // Extract Rhizome View (from rhizome.html)
  async extractRhizomeView(container) {
    try {
      const artisticData = await this.loadJSONData(
        "image_analysis/artistic_analysis_filtered.json"
      );

      if (artisticData) {
        // Create network visualization (from rhizome.html)
        this.createRhizomeNetwork(artisticData, container);
      } else {
        this.createGenericPlaceholder(
          "rhizome",
          { name: "Rhizome Network" },
          container
        );
      }
    } catch (error) {
      console.error("Error extracting rhizome view:", error);
      this.createGenericPlaceholder(
        "rhizome",
        { name: "Rhizome Network" },
        container
      );
    }
  }

  // Extract other views
  async extractTemperamentView(container) {
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
      console.error("Error extracting temperament view:", error);
      this.createGenericPlaceholder(
        "temperament",
        { name: "Temperament Scores" },
        container
      );
    }
  }

  async extractCompletenessView(container) {
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
      console.error("Error extracting completeness view:", error);
      this.createGenericPlaceholder(
        "completeness",
        { name: "Completeness Analysis" },
        container
      );
    }
  }

  async extractHashtagView(container) {
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
      console.error("Error extracting hashtag view:", error);
      this.createGenericPlaceholder(
        "hashtag",
        { name: "Hashtag Gallery" },
        container
      );
    }
  }

  async extractOpenQuestionView(container) {
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
      console.error("Error extracting open question view:", error);
      this.createGenericPlaceholder(
        "open-question",
        { name: "Open Question" },
        container
      );
    }
  }

  async extractTotalGalaxyView(container) {
    try {
      // Create galaxy visualization (from total_galaxy_accurate.html)
      this.createGalaxyVisualization(container);
    } catch (error) {
      console.error("Error extracting total galaxy view:", error);
      this.createGenericPlaceholder(
        "total-galaxy",
        { name: "Total Galaxy" },
        container
      );
    }
  }

  // Artists View Helpers (extracted from Artist.html)
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

  // Intimacy View Helpers (extracted from Intimacy.html)
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

    // Create main tunnel structure
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

  addStationMarkers(container) {
    const stations = [
      "The Beach",
      "The Park",
      "Museum",
      "Cafe",
      "Pub",
      "Class",
      "Studio",
      "Living room",
      "Bedroom",
      "Toilet",
    ];

    stations.forEach((station, index) => {
      const marker = new THREE.Mesh(
        new THREE.SphereGeometry(10, 8, 8),
        new THREE.MeshBasicMaterial({
          color: new THREE.Color().setHSL(index / stations.length, 0.7, 0.5),
          transparent: true,
          opacity: 0.8,
        })
      );

      const angle = (index / stations.length) * Math.PI * 2;
      const radius = 120;
      marker.position.set(
        Math.cos(angle) * radius,
        0,
        Math.sin(angle) * radius
      );

      container.add(marker);
    });
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

  // Rhizome View Helpers (extracted from rhizome.html)
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

    context.fillStyle = "rgba(0, 0, 0, 0.8)";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "white";
    context.font = "16px VT323";
    context.textAlign = "center";
    context.fillText(text, canvas.width / 2, canvas.height / 2 + 5);

    const texture = new THREE.CanvasTexture(canvas);
    const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
    const sprite = new THREE.Sprite(spriteMaterial);
    sprite.scale.set(50, 12.5, 1);

    return sprite;
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

    const label = this.createTextSprite(config.name);
    label.position.set(0, 100, 0);
    container.add(label);
  }

  createFallbackPlaceholder(name, config) {
    const container = new THREE.Group();
    container.position.set(
      config.position.x,
      config.position.y,
      config.position.z
    );
    container.name = `extracted-view-${name}-fallback`;

    this.createGenericPlaceholder(name, config, container);
    this.scene.add(container);
    this.loadedViews.set(name, container);
  }

  // Unload an extracted view
  async unloadView(name) {
    console.log(`Unloading extracted view: ${name}`);

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

      // Remove from tracking
      this.loadedViews.delete(name);
      this.viewContainers.delete(name);
      this.extractedContent.delete(name);

      console.log(`Successfully unloaded extracted view: ${name}`);
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
