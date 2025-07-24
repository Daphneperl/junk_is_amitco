const { minify } = require("html-minifier");
const fs = require("fs");
const path = require("path");

function minifyHtml() {
  console.log("📄 Minifying HTML files...");

  const htmlFiles = [
    "../index.html",
    "../base.html",
    "../base_w_rings.html",
    "../glitchy_eye.html",
    "../SearchFieldView.html",
    "../UI-menu.html",
  ];

  const viewDirs = ["../views"];

  // Get all HTML files from views subdirectories
  function getHtmlFiles(dir) {
    const files = [];
    if (fs.existsSync(dir)) {
      const items = fs.readdirSync(dir);
      for (const item of items) {
        const fullPath = path.join(dir, item);
        if (fs.statSync(fullPath).isDirectory()) {
          files.push(...getHtmlFiles(fullPath));
        } else if (item.endsWith(".html")) {
          files.push(fullPath);
        }
      }
    }
    return files;
  }

  const allHtmlFiles = [...htmlFiles, ...getHtmlFiles("views")];

  for (const file of allHtmlFiles) {
    if (fs.existsSync(file)) {
      try {
        const content = fs.readFileSync(file, "utf8");
        const minified = minify(content, {
          collapseWhitespace: true,
          removeComments: true,
          minifyCSS: true,
          minifyJS: true,
          removeAttributeQuotes: true,
          removeEmptyAttributes: true,
          removeOptionalTags: true,
          removeRedundantAttributes: true,
          removeScriptTypeAttributes: true,
          removeStyleLinkTypeAttributes: true,
          useShortDoctype: true,
        });

        fs.writeFileSync(file, minified);
        console.log(`✓ Minified ${file}`);
      } catch (error) {
        console.error(`✗ Error minifying ${file}:`, error.message);
      }
    }
  }

  console.log("✅ HTML minification complete!");
}

minifyHtml();
