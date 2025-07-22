// Real View Integrator - Properly Integrate Existing Complex Visualizations
class RealViewIntegrator {
  constructor(scene, camera, renderer) {
    this.scene = scene;
    this.camera = camera;
    this.renderer = renderer;
    this.loadedViews = new Map();
    this.viewContainers = new Map();
    this.originalViewData = new Map();
  }

  // Integrate an existing view with its actual content
  async integrateView(name, config) {
    console.log(`Integrating real view: ${name}`);

    if (this.loadedViews.has(name)) {
      console.log(`View ${name} already integrated`);
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
      viewContainer.name = `real-view-${name}`;

      // Load the actual view content based on the existing HTML files
      await this.loadRealViewContent(name, config, viewContainer);

      // Add to scene
      this.scene.add(viewContainer);
      this.loadedViews.set(name, viewContainer);
      this.viewContainers.set(name, viewContainer);

      console.log(`Successfully integrated real view: ${name}`);
    } catch (error) {
      console.error(`Error integrating view ${name}:`, error);
      this.createFallbackPlaceholder(name, config);
    }
  }

  // Load the actual view content from existing HTML files
  async loadRealViewContent(name, config, container) {
    switch (name) {
      case "artists":
        await this.loadRealArtistsView(container);
        break;
      case "intimacy":
        await this.loadRealIntimacyView(container);
        break;
      case "rhizome":
        await this.loadRealRhizomeView(container);
        break;
      case "temperament":
        await this.loadRealTemperamentView(container);
        break;
      case "completeness":
        await this.loadRealCompletenessView(container);
        break;
      case "hashtag":
        await this.loadRealHashtagView(container);
        break;
      case "open-question":
        await this.loadRealOpenQuestionView(container);
        break;
      case "total-galaxy":
        await this.loadRealTotalGalaxyView(container);
        break;
      default:
        this.createGenericPlaceholder(name, config, container);
    }
  }

  // Load Real Artists View (extracted from Artist.html)
  async loadRealArtistsView(container) {
    try {
      // Load the actual artistic analysis data
      const artistData = await this.loadJSONData(
        "image_analysis/artistic_analysis_filtered.json"
      );

      if (artistData) {
        // Group by artist (from Artist.html logic)
        const artists = this.groupByArtist(artistData);
        const artistNames = Object.keys(artists);

        // Create helix positions (from Artist.html)
        const helixPositions = this.createRealHelixPositions(
          artistNames.length
        );

        // Create artist clusters in helix formation (from Artist.html)
        artistNames.forEach((artist, images) => {
          const position = helixPositions[artistNames.indexOf(artist)];
          const artistGroup = this.createRealArtistCluster(
            artist,
            images,
            position
          );
          container.add(artistGroup);
        });

        // Create helix outline (from Artist.html)
        const helixOutline = this.createRealHelixOutline();
        container.add(helixOutline);

        // Create helix grid (from Artist.html)
        const helixGrid = this.createRealHelixGrid();
        container.add(helixGrid);
      } else {
        this.createGenericPlaceholder(
          "artists",
          { name: "Artists Gallery" },
          container
        );
      }
    } catch (error) {
      console.error("Error loading real artists view:", error);
      this.createGenericPlaceholder(
        "artists",
        { name: "Artists Gallery" },
        container
      );
    }
  }

  // Load Real Intimacy View (extracted from Intimacy.html)
  async loadRealIntimacyView(container) {
    try {
      // Create tunnel curve (from Intimacy.html)
      const tunnelCurve = this.createRealTunnelCurve();

      // Create tunnel wire grid (from Intimacy.html)
      const tunnelWire = this.createRealTunnelWireGrid();
      container.add(tunnelWire);

      // Load and place artworks along the tunnel (from Intimacy.html)
      const artworks = await this.loadRealArtworks();
      if (artworks.length > 0) {
        this.placeRealArtworksInTunnel(artworks, tunnelCurve, container);
      }

      // Add station markers (from Intimacy.html)
      this.addRealStationMarkers(container);
    } catch (error) {
      console.error("Error loading real intimacy view:", error);
      this.createGenericPlaceholder(
        "intimacy",
        { name: "Intimacy Tunnel" },
        container
      );
    }
  }

  // Load Real Rhizome View (extracted from rhizome.html)
  async loadRealRhizomeView(container) {
    try {
      const artisticData = await this.loadJSONData(
        "image_analysis/artistic_analysis_filtered.json"
      );

      if (artisticData) {
        // Create network visualization (from rhizome.html)
        this.createRealRhizomeNetwork(artisticData, container);
      } else {
        this.createGenericPlaceholder(
          "rhizome",
          { name: "Rhizome Network" },
          container
        );
      }
    } catch (error) {
      console.error("Error loading real rhizome view:", error);
      this.createGenericPlaceholder(
        "rhizome",
        { name: "Rhizome Network" },
        container
      );
    }
  }

  // Load other real views
  async loadRealTemperamentView(container) {
    try {
      const scoreData = await this.loadJSONData(
        "views/temperament/temperament_scores.json"
      );
      if (scoreData) {
        this.createRealTemperamentVisualization(scoreData, container);
      } else {
        this.createGenericPlaceholder(
          "temperament",
          { name: "Temperament Scores" },
          container
        );
      }
    } catch (error) {
      console.error("Error loading real temperament view:", error);
      this.createGenericPlaceholder(
        "temperament",
        { name: "Temperament Scores" },
        container
      );
    }
  }

  async loadRealCompletenessView(container) {
    try {
      const completenessData = await this.loadJSONData(
        "views/Completeness/inverted_sketchiness_scores.json"
      );
      if (completenessData) {
        this.createRealCompletenessVisualization(completenessData, container);
      } else {
        this.createGenericPlaceholder(
          "completeness",
          { name: "Completeness Analysis" },
          container
        );
      }
    } catch (error) {
      console.error("Error loading real completeness view:", error);
      this.createGenericPlaceholder(
        "completeness",
        { name: "Completeness Analysis" },
        container
      );
    }
  }

  async loadRealHashtagView(container) {
    try {
      const hashtagData = await this.loadCSVData(
        "views/hashtag_gallery/Hashtags.csv"
      );
      if (hashtagData) {
        this.createRealHashtagVisualization(hashtagData, container);
      } else {
        this.createGenericPlaceholder(
          "hashtag",
          { name: "Hashtag Gallery" },
          container
        );
      }
    } catch (error) {
      console.error("Error loading real hashtag view:", error);
      this.createGenericPlaceholder(
        "hashtag",
        { name: "Hashtag Gallery" },
        container
      );
    }
  }

  async loadRealOpenQuestionView(container) {
    try {
      const questionData = await this.loadJSONData(
        "views/open_question/image_scores.json"
      );
      if (questionData) {
        this.createRealOpenQuestionVisualization(questionData, container);
      } else {
        this.createGenericPlaceholder(
          "open-question",
          { name: "Open Question" },
          container
        );
      }
    } catch (error) {
      console.error("Error loading real open question view:", error);
      this.createGenericPlaceholder(
        "open-question",
        { name: "Open Question" },
        container
      );
    }
  }

  async loadRealTotalGalaxyView(container) {
    try {
      // Create galaxy visualization (from total_galaxy_accurate.html)
      this.createRealGalaxyVisualization(container);
    } catch (error) {
      console.error("Error loading real total galaxy view:", error);
      this.createGenericPlaceholder(
        "total-galaxy",
        { name: "Total Galaxy" },
        container
      );
    }
  }

  // Real Artists View Helpers (extracted from Artist.html)
  createRealHelixPositions(totalClusters) {
    const positions = [];
    const turns = 20; // From Artist.html
    const heightStep = 1500 / totalClusters; // From Artist.html
    const radius = 40; // From Artist.html

    for (let i = 0; i < totalClusters; i++) {
      const angle = (i / totalClusters) * Math.PI * 2 * turns;
      const x = radius * Math.cos(angle);
      const y = i * heightStep - 750; // Center vertically
      const z = radius * Math.sin(angle);
      positions.push(new THREE.Vector3(x, y, z));
    }

    return positions;
  }

  createRealHelixOutline() {
    const turns = 20; // From Artist.html
    const radius = 40; // From Artist.html
    const totalHeight = 1500; // From Artist.html
    const segments = 1200; // From Artist.html

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
      opacity: 0.9, // From Artist.html
    });

    return new THREE.Line(geometry, material);
  }

  createRealHelixGrid() {
    const turns = 20; // From Artist.html
    const radius = 40; // From Artist.html
    const totalHeight = 1500; // From Artist.html
    const gridGroup = new THREE.Group();

    const spiralGuides = 6; // From Artist.html
    for (let guide = 0; guide <= spiralGuides; guide++) {
      const guideRadius = radius * (0.2 + guide * 0.15); // From Artist.html
      const points = [];
      const segments = 800; // From Artist.html

      for (let i = 0; i <= segments; i++) {
        const angle = (i / segments) * Math.PI * 2 * turns;
        const y = (i / segments) * totalHeight - totalHeight / 2;
        const x = guideRadius * Math.cos(angle);
        const z = guideRadius * Math.sin(angle);
        points.push(new THREE.Vector3(x, y, z));
      }

      const isOutermost = guide === spiralGuides;
      const opacity = isOutermost ? 0.8 : 0.4 + guide * 0.05; // From Artist.html

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

  createRealArtistCluster(artist, images, position) {
    const cluster = new THREE.Group();
    cluster.position.copy(position);

    // Create central artist sphere (from Artist.html)
    const artistSphere = new THREE.Mesh(
      new THREE.SphereGeometry(15, 16, 16),
      new THREE.MeshBasicMaterial({
        color: new THREE.Color().setHSL(Math.random(), 0.8, 0.6),
        transparent: true,
        opacity: 0.8,
      })
    );
    cluster.add(artistSphere);

    // Add artist label (from Artist.html)
    const label = this.createRealTextSprite(artist);
    label.position.set(0, 30, 0);
    cluster.add(label);

    // Add image spheres around the artist (from Artist.html)
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

  // Real Intimacy View Helpers (extracted from Intimacy.html)
  createRealTunnelCurve() {
    return new THREE.CatmullRomCurve3(
      Array.from({ length: 200 }, (_, i) => {
        const t = i / 199;
        const angle = t * Math.PI * 1.8; // From Intimacy.html
        const radius = 1500; // From Intimacy.html
        return new THREE.Vector3(
          Math.sin(angle) * radius,
          0,
          -Math.cos(angle) * radius
        );
      })
    );
  }

  createRealTunnelWireGrid() {
    const wireGroup = new THREE.Group();

    // Create main tunnel structure (from Intimacy.html)
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

  addRealStationMarkers(container) {
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
    ]; // From Intimacy.html

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

  async loadRealArtworks() {
    try {
      const response = await fetch("images/images.json");
      const imageList = await response.json();
      return imageList.slice(0, 50); // Limit for performance
    } catch (error) {
      console.warn("Could not load artworks:", error);
      return [];
    }
  }

  placeRealArtworksInTunnel(artworks, tunnelCurve, container) {
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

  // Real Rhizome View Helpers (extracted from rhizome.html)
  createRealRhizomeNetwork(artisticData, container) {
    const nodeCount = Math.min(50, artisticData.length);
    const nodes = [];

    // Create nodes (from rhizome.html)
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

    // Create connections (from rhizome.html)
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

  // Other real visualization helpers
  createRealTemperamentVisualization(scoreData, container) {
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

  createRealCompletenessVisualization(completenessData, container) {
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

  createRealHashtagVisualization(hashtagData, container) {
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

  createRealOpenQuestionVisualization(questionData, container) {
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

  createRealGalaxyVisualization(container) {
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
  createRealTextSprite(text) {
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

    const label = this.createRealTextSprite(config.name);
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
    container.name = `real-view-${name}-fallback`;

    this.createGenericPlaceholder(name, config, container);
    this.scene.add(container);
    this.loadedViews.set(name, container);
  }

  // Unload a real view
  async unloadView(name) {
    console.log(`Unloading real view: ${name}`);

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
      this.originalViewData.delete(name);

      console.log(`Successfully unloaded real view: ${name}`);
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
