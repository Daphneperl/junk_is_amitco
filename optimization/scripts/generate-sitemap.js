const fs = require("fs").promises;
const path = require("path");

async function generateSitemap() {
  console.log("🗺️  Generating sitemap...");

  const baseUrl = process.env.VERCEL_URL || "https://your-domain.vercel.app";
  const sitemap = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
  ];

  // Add main pages
  const mainPages = [
    "",
    "/index.html",
    "/base.html",
    "/base_w_rings.html",
    "/glitchy_eye.html",
    "/SearchFieldView.html",
    "/UI-menu.html",
  ];

  for (const page of mainPages) {
    sitemap.push(
      "  <url>",
      `    <loc>${baseUrl}${page}</loc>`,
      "    <lastmod>" + new Date().toISOString() + "</lastmod>",
      "    <changefreq>weekly</changefreq>",
      "    <priority>0.8</priority>",
      "  </url>"
    );
  }

  // Add view pages
  const viewDirs = ["../views"];

  for (const viewDir of viewDirs) {
    if (await directoryExists(viewDir)) {
      const htmlFiles = await findHtmlFiles(viewDir);

      for (const htmlFile of htmlFiles) {
        const relativePath = htmlFile.replace(/\\/g, "/");
        sitemap.push(
          "  <url>",
          `    <loc>${baseUrl}/${relativePath}</loc>`,
          "    <lastmod>" + new Date().toISOString() + "</lastmod>",
          "    <changefreq>monthly</changefreq>",
          "    <priority>0.6</priority>",
          "  </url>"
        );
      }
    }
  }

  sitemap.push("</urlset>");

  await fs.writeFile("../public/sitemap.xml", sitemap.join("\n"));
  console.log("✅ Sitemap generated: ../public/sitemap.xml");
}

async function directoryExists(dirPath) {
  try {
    await fs.access(dirPath);
    return true;
  } catch {
    return false;
  }
}

async function findHtmlFiles(dir) {
  const files = [];

  async function scanDirectory(currentDir) {
    const items = await fs.readdir(currentDir);

    for (const item of items) {
      const fullPath = path.join(currentDir, item);
      const stat = await fs.stat(fullPath);

      if (stat.isDirectory()) {
        await scanDirectory(fullPath);
      } else if (item.endsWith(".html")) {
        files.push(fullPath);
      }
    }
  }

  await scanDirectory(dir);
  return files;
}

// Create public directory if it doesn't exist
async function ensurePublicDir() {
  try {
    await fs.access("../public");
  } catch {
    await fs.mkdir("../public");
  }
}

async function main() {
  await ensurePublicDir();
  await generateSitemap();
}

main().catch(console.error);
