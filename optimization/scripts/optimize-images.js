const imagemin = require("imagemin");
const imageminMozjpeg = require("imagemin-mozjpeg");
const imageminPngquant = require("imagemin-pngquant");
const imageminWebp = require("imagemin-webp");
const fs = require("fs");
const path = require("path");

async function optimizeImages() {
  console.log("🖼️  Optimizing images...");

  const imageDirs = ["images", "assets", "Eyes", "UploadButton"];

  for (const dir of imageDirs) {
    if (fs.existsSync(dir)) {
      console.log(`Processing ${dir}...`);

      const files = await imagemin([`${dir}/*.{jpg,jpeg,png,gif}`], {
        destination: dir,
        plugins: [
          imageminMozjpeg({ quality: 80 }),
          imageminPngquant({ quality: [0.6, 0.8] }),
          imageminWebp({ quality: 80 }),
        ],
      });

      console.log(`✓ Optimized ${files.length} files in ${dir}`);
    }
  }

  console.log("✅ Image optimization complete!");
}

optimizeImages().catch(console.error);
