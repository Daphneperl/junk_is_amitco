/**
 * Progressive Image Loader
 * Implements lazy loading, blur-up technique, and optimal format selection
 */

class ProgressiveImageLoader {
  constructor(options = {}) {
    this.options = {
      rootMargin: "50px",
      threshold: 0.1,
      placeholderSize: 20,
      ...options,
    };

    this.observer = null;
    this.loadedImages = new Set();
    this.imageManifest = null;
    this.init();
  }

  async init() {
    // Load image manifest
    try {
      const response = await fetch("/image-manifest.json");
      this.imageManifest = await response.json();
    } catch (error) {
      console.warn("Image manifest not found, using fallback loading");
    }

    // Initialize intersection observer for lazy loading
    this.initIntersectionObserver();
  }

  initIntersectionObserver() {
    this.observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            this.loadImage(entry.target);
            this.observer.unobserve(entry.target);
          }
        });
      },
      {
        rootMargin: this.options.rootMargin,
        threshold: this.options.threshold,
      }
    );
  }

  async loadImage(imgElement) {
    const filename = imgElement.dataset.src || imgElement.src;
    if (!filename || this.loadedImages.has(filename)) return;

    this.loadedImages.add(filename);

    try {
      // Show placeholder first
      await this.showPlaceholder(imgElement, filename);

      // Load optimal format
      const optimalSrc = await this.getOptimalImageSource(filename);

      // Load full image
      await this.loadFullImage(imgElement, optimalSrc);
    } catch (error) {
      console.error(`Error loading image ${filename}:`, error);
      this.showFallback(imgElement);
    }
  }

  async showPlaceholder(imgElement, filename) {
    const placeholderSrc = this.getPlaceholderSrc(filename);

    if (placeholderSrc) {
      // Create blur effect
      imgElement.style.filter = "blur(10px)";
      imgElement.style.transform = "scale(1.1)";
      imgElement.style.transition = "filter 0.3s ease, transform 0.3s ease";

      // Load tiny placeholder
      const placeholder = new Image();
      placeholder.onload = () => {
        imgElement.src = placeholderSrc;
      };
      placeholder.src = placeholderSrc;
    }
  }

  async loadFullImage(imgElement, src) {
    return new Promise((resolve, reject) => {
      const fullImage = new Image();

      fullImage.onload = () => {
        // Smooth transition from placeholder to full image
        imgElement.style.filter = "blur(0px)";
        imgElement.style.transform = "scale(1)";
        imgElement.src = src;
        resolve();
      };

      fullImage.onerror = reject;
      fullImage.src = src;
    });
  }

  getPlaceholderSrc(filename) {
    if (!this.imageManifest) return null;

    const baseName = this.getBaseName(filename);
    const thumbnailPath = this.imageManifest.thumbnails[baseName];

    if (thumbnailPath) {
      return thumbnailPath;
    }

    // Fallback: generate tiny data URL
    return this.generateTinyPlaceholder(filename);
  }

  async getOptimalImageSource(filename) {
    const baseName = this.getBaseName(filename);

    // Check for WebP support
    if (this.supportsWebP()) {
      const webpPath = this.imageManifest?.webp[baseName];
      if (webpPath) return webpPath;
    }

    // Return original optimized path
    return this.imageManifest?.images[baseName] || filename;
  }

  getBaseName(filename) {
    return filename.replace(/\.[^/.]+$/, "");
  }

  supportsWebP() {
    const canvas = document.createElement("canvas");
    canvas.width = 1;
    canvas.height = 1;
    return canvas.toDataURL("image/webp").indexOf("data:image/webp") === 0;
  }

  generateTinyPlaceholder(filename) {
    // Create a tiny colored placeholder
    const canvas = document.createElement("canvas");
    canvas.width = this.options.placeholderSize;
    canvas.height = this.options.placeholderSize;

    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#f0f0f0";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    return canvas.toDataURL();
  }

  showFallback(imgElement) {
    imgElement.style.filter = "grayscale(100%) opacity(50%)";
    imgElement.style.background = "#f0f0f0";
  }

  // Public API for manual image loading
  observe(imgElement) {
    if (this.observer) {
      this.observer.observe(imgElement);
    }
  }

  unobserve(imgElement) {
    if (this.observer) {
      this.observer.unobserve(imgElement);
    }
  }

  // Batch load images for critical content
  async preloadCritical(images) {
    const promises = images.map((filename) => {
      return new Promise((resolve) => {
        const img = new Image();
        img.onload = resolve;
        img.onerror = resolve;
        img.src = this.getOptimalImageSource(filename);
      });
    });

    await Promise.all(promises);
  }
}

// Three.js Texture Loader enhancement
class ProgressiveTextureLoader {
  constructor() {
    this.loader = new THREE.TextureLoader();
    this.progressiveLoader = new ProgressiveImageLoader();
  }

  load(url, onLoad, onProgress, onError) {
    // First load a tiny placeholder
    const placeholderUrl = this.progressiveLoader.getPlaceholderSrc(url);

    if (placeholderUrl) {
      this.loader.load(
        placeholderUrl,
        (placeholderTexture) => {
          // Create a temporary texture with placeholder
          const tempTexture = placeholderTexture.clone();
          tempTexture.minFilter = THREE.LinearFilter;
          tempTexture.magFilter = THREE.LinearFilter;

          // Call onLoad with placeholder
          if (onLoad) onLoad(tempTexture);

          // Load full resolution texture
          this.loader.load(
            url,
            (fullTexture) => {
              // Replace placeholder with full texture
              tempTexture.image = fullTexture.image;
              tempTexture.needsUpdate = true;

              // Update texture properties
              tempTexture.minFilter = THREE.LinearMipmapLinearFilter;
              tempTexture.magFilter = THREE.LinearFilter;
              tempTexture.generateMipmaps = true;

              if (onLoad) onLoad(tempTexture);
            },
            onProgress,
            onError
          );
        },
        onProgress,
        onError
      );
    } else {
      // Fallback to normal loading
      this.loader.load(url, onLoad, onProgress, onError);
    }
  }
}

// Export for use in other scripts
if (typeof module !== "undefined" && module.exports) {
  module.exports = { ProgressiveImageLoader, ProgressiveTextureLoader };
} else {
  window.ProgressiveImageLoader = ProgressiveImageLoader;
  window.ProgressiveTextureLoader = ProgressiveTextureLoader;
}
