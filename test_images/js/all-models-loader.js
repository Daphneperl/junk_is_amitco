// All Models Loader - Load and Display All Actual Models from Existing Views
class AllModelsLoader {
  constructor(scene, camera, renderer) {
    this.scene = scene;
    this.camera = camera;
    this.renderer = renderer;
    this.loadedModels = new Map();
    this.modelContainers = new Map();
    this.textureLoader = new THREE.TextureLoader();
  }

  // Load all models from existing views
  async loadAllModels() {
    console.log("Loading all models from existing views...");

    // Load all models simultaneously
    const loadPromises = [
      this.loadArtistsModel(),
      this.loadIntimacyModel(),
      this.loadRhizomeModel(),
      this.loadTemperamentModel(),
      this.loadCompletenessModel(),
      this.loadHashtagModel(),
      this.loadOpenQuestionModel(),
      this.loadTotalGalaxyModel(),
    ];

    await Promise.all(loadPromises);
    console.log("All models loaded successfully!");
  }

  // Load Artists Model (from Artist.html)
  async loadArtistsModel() {
    try {
      const container = new THREE.Group();
      container.name = "artists-model";
      container.position.set(0, 0, 0);

      // Load artistic analysis data
      const artistData = await this.loadJSONData(
        "image_analysis/artistic_analysis_filtered.json"
      );

      if (artistData) {
        // Group by artist
        const artists = this.groupByArtist(artistData);
        const artistNames = Object.keys(artists);

        // Create helix positions (from Artist.html)
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
      }

      this.scene.add(container);
      this.loadedModels.set("artists", container);
      this.modelContainers.set("artists", container);
    } catch (error) {
      console.error("Error loading artists model:", error);
    }
  }

  // Load Intimacy Model (from Intimacy.html)
  async loadIntimacyModel() {
    try {
      const container = new THREE.Group();
      container.name = "intimacy-model";
      container.position.set(500, 0, 0);

      // Create tunnel curve (from Intimacy.html)
      const tunnelCurve = this.createTunnelCurve();

      // Create tunnel wire grid (from Intimacy.html)
      const tunnelWire = this.createTunnelWireGrid(tunnelCurve);
      container.add(tunnelWire);

      // Load and place artworks along the tunnel
      const artworks = await this.loadArtworks();
      if (artworks.length > 0) {
        this.placeArtworksInTunnel(artworks, tunnelCurve, container);
      }

      // Add station markers
      this.addStationMarkers(container);

      this.scene.add(container);
      this.loadedModels.set("intimacy", container);
      this.modelContainers.set("intimacy", container);
    } catch (error) {
      console.error("Error loading intimacy model:", error);
    }
  }

  // Load Rhizome Model (from rhizome.html)
  async loadRhizomeModel() {
    try {
      const container = new THREE.Group();
      container.name = "rhizome-model";
      container.position.set(0, 0, 500);

      const artisticData = await this.loadJSONData(
        "image_analysis/artistic_analysis_filtered.json"
      );

      if (artisticData) {
        // Create network visualization (from rhizome.html)
        this.createRhizomeNetwork(artisticData, container);
      }

      this.scene.add(container);
      this.loadedModels.set("rhizome", container);
      this.modelContainers.set("rhizome", container);
    } catch (error) {
      console.error("Error loading rhizome model:", error);
    }
  }

  // Load Temperament Model
  async loadTemperamentModel() {
    try {
      const container = new THREE.Group();
      container.name = "temperament-model";
      container.position.set(500, 0, 500);

      const scoreData = await this.loadJSONData(
        "views/temperament/temperament_scores.json"
      );
      if (scoreData) {
        this.createTemperamentVisualization(scoreData, container);
      }

      this.scene.add(container);
      this.loadedModels.set("temperament", container);
      this.modelContainers.set("temperament", container);
    } catch (error) {
      console.error("Error loading temperament model:", error);
    }
  }

  // Load Completeness Model
  async loadCompletenessModel() {
    try {
      const container = new THREE.Group();
      container.name = "completeness-model";
      container.position.set(0, 500, 0);

      const completenessData = await this.loadJSONData(
        "views/Completeness/inverted_sketchiness_scores.json"
      );
      if (completenessData) {
        this.createCompletenessVisualization(completenessData, container);
      }

      this.scene.add(container);
      this.loadedModels.set("completeness", container);
      this.modelContainers.set("completeness", container);
    } catch (error) {
      console.error("Error loading completeness model:", error);
    }
  }

  // Load Hashtag Model
  async loadHashtagModel() {
    try {
      const container = new THREE.Group();
      container.name = "hashtag-model";
      container.position.set(500, 500, 0);

      const hashtagData = await this.loadCSVData(
        "views/hashtag_gallery/Hashtags.csv"
      );
      if (hashtagData) {
        this.createHashtagVisualization(hashtagData, container);
      }

      this.scene.add(container);
      this.loadedModels.set("hashtag", container);
      this.modelContainers.set("hashtag", container);
    } catch (error) {
      console.error("Error loading hashtag model:", error);
    }
  }

  // Load Open Question Model
  async loadOpenQuestionModel() {
    try {
      const container = new THREE.Group();
      container.name = "open-question-model";
      container.position.set(0, 500, 500);

      const questionData = await this.loadJSONData(
        "views/open_question/image_scores.json"
      );
      if (questionData) {
        this.createOpenQuestionVisualization(questionData, container);
      }

      this.scene.add(container);
      this.loadedModels.set("open-question", container);
      this.modelContainers.set("open-question", container);
    } catch (error) {
      console.error("Error loading open question model:", error);
    }
  }

  // Load Total Galaxy Model
  async loadTotalGalaxyModel() {
    try {
      const container = new THREE.Group();
      container.name = "total-galaxy-model";
      container.position.set(500, 500, 500);

      // Create galaxy visualization
      this.createGalaxyVisualization(container);

      this.scene.add(container);
      this.loadedModels.set("total-galaxy", container);
      this.modelContainers.set("total-galaxy", container);
    } catch (error) {
      console.error("Error loading total galaxy model:", error);
    }
  }

  // Artists Model Helpers (from Artist.html)
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

  // Intimacy Model Helpers (from Intimacy.html)
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

  createTunnelWireGrid(tunnelCurve) {
    const wireGroup = new THREE.Group();
    const tunnelRadius = 100;

    // Create main tunnel structure
    const tunnelGeometry = new THREE.CylinderGeometry(
      tunnelRadius,
      tunnelRadius,
      800,
      32
    );
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
      return imageList.slice(0, 50);
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

  // Rhizome Model Helpers (from rhizome.html)
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

  // Other Model Helpers
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
