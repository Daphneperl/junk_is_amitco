const sharp = require("sharp");
const fs = require("fs").promises;
const path = require("path");
const imagemin = require("imagemin");
const imageminMozjpeg = require("imagemin-mozjpeg");
const imageminPngquant = require("imagemin-pngquant");
const imageminWebp = require("imagemin-webp");

class AdvancedImageOptimizer {
  constructor() {
    this.sizes = [20, 100, 300, 600, 1200, 2400]; // Thumbnail to full size
    this.quality = {
      jpeg: 80,
      webp: 85,
      png: 90,
    };
    this.imageDirs = ["../images", "../assets", "../Eyes", "../UploadButton"];
  }

  async optimizeImages() {
    console.log("🖼️  Starting advanced image optimization...");

    for (const dir of this.imageDirs) {
      if (await this.directoryExists(dir)) {
        console.log(`Processing ${dir}...`);
        await this.processDirectory(dir);
      }
    }

    console.log("✅ Advanced image optimization complete!");
  }

  async directoryExists(dirPath) {
    try {
      await fs.access(dirPath);
      return true;
    } catch {
      return false;
    }
  }

  async processDirectory(dirPath) {
    const files = await fs.readdir(dirPath);
    const imageFiles = files.filter((file) =>
      /\.(jpg|jpeg|png|gif|webp)$/i.test(file)
    );

    console.log(`Found ${imageFiles.length} images in ${dirPath}`);

    for (const file of imageFiles) {
      const filePath = path.join(dirPath, file);
      await this.processImage(filePath);
    }
  }

  async processImage(filePath) {
    try {
      const ext = path.extname(filePath).toLowerCase();
      const baseName = path.basename(filePath, ext);
      const dir = path.dirname(filePath);

      // Skip if already processed
      if (baseName.includes("_thumb") || baseName.includes("_webp")) {
        return;
      }

      console.log(`Processing: ${filePath}`);

      // Generate multiple sizes
      await this.generateSizes(filePath, dir, baseName, ext);

      // Generate WebP versions
      await this.generateWebP(filePath, dir, baseName);

      // Optimize original
      await this.optimizeOriginal(filePath, ext);
    } catch (error) {
      console.error(`Error processing ${filePath}:`, error.message);
    }
  }

  async generateSizes(filePath, dir, baseName, ext) {
    const image = sharp(filePath);
    const metadata = await image.metadata();

    for (const size of this.sizes) {
      if (metadata.width <= size) continue; // Skip if image is smaller

      const outputPath = path.join(dir, `${baseName}_${size}${ext}`);

      await image
        .resize(size, null, {
          withoutEnlargement: true,
          fit: "inside",
        })
        .jpeg({ quality: this.quality.jpeg })
        .toFile(outputPath);

      console.log(`  Generated: ${outputPath}`);
    }
  }

  async generateWebP(filePath, dir, baseName) {
    const image = sharp(filePath);
    const metadata = await image.metadata();

    // Generate WebP versions for different sizes
    for (const size of this.sizes) {
      if (metadata.width <= size) continue;

      const outputPath = path.join(dir, `${baseName}_${size}.webp`);

      await image
        .resize(size, null, {
          withoutEnlargement: true,
          fit: "inside",
        })
        .webp({ quality: this.quality.webp })
        .toFile(outputPath);

      console.log(`  Generated WebP: ${outputPath}`);
    }

    // Generate full-size WebP
    const fullWebPPath = path.join(dir, `${baseName}.webp`);
    await image.webp({ quality: this.quality.webp }).toFile(fullWebPPath);

    console.log(`  Generated full WebP: ${fullWebPPath}`);
  }

  async optimizeOriginal(filePath, ext) {
    const optimizedPath = filePath.replace(ext, `_optimized${ext}`);

    if (ext === ".jpg" || ext === ".jpeg") {
      await imagemin([filePath], {
        destination: path.dirname(filePath),
        plugins: [imageminMozjpeg({ quality: this.quality.jpeg })],
      });
    } else if (ext === ".png") {
      await imagemin([filePath], {
        destination: path.dirname(filePath),
        plugins: [
          imageminPngquant({
            quality: [this.quality.png / 100, this.quality.png / 100],
          }),
        ],
      });
    }

    console.log(`  Optimized: ${filePath}`);
  }

  async generateImageManifest() {
    console.log("📝 Generating image manifest...");

    const manifest = {
      images: {},
      thumbnails: {},
      webp: {},
    };

    for (const dir of this.imageDirs) {
      if (await this.directoryExists(dir)) {
        const files = await fs.readdir(dir);

        for (const file of files) {
          const filePath = path.join(dir, file);
          const stats = await fs.stat(filePath);

          if (stats.isFile() && /\.(jpg|jpeg|png|gif|webp)$/i.test(file)) {
            const baseName = path.basename(file, path.extname(file));

            if (file.includes("_20")) {
              manifest.thumbnails[baseName.replace("_20", "")] = filePath;
            } else if (file.includes(".webp")) {
              manifest.webp[baseName] = filePath;
            } else if (!file.includes("_") && !file.includes("optimized")) {
              manifest.images[baseName] = filePath;
            }
          }
        }
      }
    }

    await fs.writeFile(
      "../image-manifest.json",
      JSON.stringify(manifest, null, 2)
    );
    console.log("✅ Image manifest generated: ../image-manifest.json");
  }
}

// Run optimization
async function main() {
  const optimizer = new AdvancedImageOptimizer();

  console.log("🚀 Starting advanced image optimization...");
  console.log(
    "This will create multiple sizes and formats for progressive loading"
  );

  await optimizer.optimizeImages();
  await optimizer.generateImageManifest();

  console.log("🎉 All optimizations complete!");
  console.log("📊 Check ../image-manifest.json for optimized image references");
}

main().catch(console.error);
