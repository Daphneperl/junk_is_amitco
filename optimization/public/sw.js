const CACHE_NAME = "junk-is-amitco-v2";
const STATIC_CACHE = "static-v2";
const IMAGE_CACHE = "images-v2";
const API_CACHE = "api-v2";

// Files to cache immediately
const STATIC_FILES = [
  "/",
  "/index.html",
  "/base.html",
  "/base_w_rings.html",
  "/glitchy_eye.html",
  "/SearchFieldView.html",
  "/UI-menu.html",
  "/assets/VT323-Regular.ttf",
  "/assets/Heming.ttf",
  "/assets/pixel.ttf",
  "https://d3js.org/d3.v7.min.js",
  "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js",
  "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js",
  "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/EffectComposer.js",
  "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/RenderPass.js",
  "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/postprocessing/ShaderPass.js",
  "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/shaders/CopyShader.js",
];

// Install event - cache static files
self.addEventListener("install", (event) => {
  console.log("Service Worker installing...");

  event.waitUntil(
    Promise.all([
      caches.open(STATIC_CACHE).then((cache) => {
        console.log("Caching static files...");
        return cache.addAll(STATIC_FILES);
      }),
      caches.open(IMAGE_CACHE).then((cache) => {
        console.log("Image cache ready");
        return cache;
      }),
      caches.open(API_CACHE).then((cache) => {
        console.log("API cache ready");
        return cache;
      }),
    ]).then(() => {
      console.log("Service Worker installed successfully");
      return self.skipWaiting();
    })
  );
});

// Activate event - clean up old caches
self.addEventListener("activate", (event) => {
  console.log("Service Worker activating...");

  event.waitUntil(
    caches
      .keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames.map((cacheName) => {
            if (
              cacheName !== STATIC_CACHE &&
              cacheName !== IMAGE_CACHE &&
              cacheName !== API_CACHE
            ) {
              console.log("Deleting old cache:", cacheName);
              return caches.delete(cacheName);
            }
          })
        );
      })
      .then(() => {
        console.log("Service Worker activated");
        return self.clients.claim();
      })
  );
});

// Fetch event - serve from cache or network
self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip non-GET requests
  if (request.method !== "GET") return;

  // Handle different types of requests
  if (isStaticFile(url.pathname)) {
    event.respondWith(handleStaticFile(request));
  } else if (isImageFile(url.pathname)) {
    event.respondWith(handleImageFile(request));
  } else if (isApiRequest(url.pathname)) {
    event.respondWith(handleApiRequest(request));
  } else {
    event.respondWith(handleDefaultRequest(request));
  }
});

function isStaticFile(pathname) {
  return (
    STATIC_FILES.some((file) => pathname.includes(file)) ||
    pathname.endsWith(".html") ||
    pathname.endsWith(".js") ||
    pathname.endsWith(".css") ||
    pathname.endsWith(".ttf") ||
    pathname.endsWith(".woff") ||
    pathname.endsWith(".woff2")
  );
}

function isImageFile(pathname) {
  return (
    pathname.includes("/images/") ||
    pathname.includes("/assets/") ||
    pathname.includes("/Eyes/") ||
    pathname.includes("/UploadButton/") ||
    /\.(jpg|jpeg|png|gif|webp|avif)$/i.test(pathname)
  );
}

function isApiRequest(pathname) {
  return (
    pathname.includes("/api/") ||
    pathname.includes("/search") ||
    pathname.includes("/scores") ||
    pathname.includes("/health")
  );
}

async function handleStaticFile(request) {
  try {
    // Try cache first
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }

    // Fallback to network
    const networkResponse = await fetch(request);

    // Cache successful responses
    if (networkResponse.ok) {
      const cache = await caches.open(STATIC_CACHE);
      cache.put(request, networkResponse.clone());
    }

    return networkResponse;
  } catch (error) {
    console.error("Static file fetch failed:", error);
    return new Response("Offline - Static file not available", { status: 503 });
  }
}

async function handleImageFile(request) {
  try {
    // Try cache first
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }

    // Try network
    const networkResponse = await fetch(request);

    if (networkResponse.ok) {
      // Cache successful image responses
      const cache = await caches.open(IMAGE_CACHE);
      cache.put(request, networkResponse.clone());
    }

    return networkResponse;
  } catch (error) {
    console.error("Image fetch failed:", error);

    // Return a placeholder image
    return new Response(
      `<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
        <rect width="100" height="100" fill="#f0f0f0"/>
        <text x="50" y="50" text-anchor="middle" fill="#999">Image</text>
      </svg>`,
      {
        headers: { "Content-Type": "image/svg+xml" },
      }
    );
  }
}

async function handleApiRequest(request) {
  try {
    // For API requests, try network first, then cache
    const networkResponse = await fetch(request);

    if (networkResponse.ok) {
      // Cache successful API responses
      const cache = await caches.open(API_CACHE);
      cache.put(request, networkResponse.clone());
    }

    return networkResponse;
  } catch (error) {
    console.error("API request failed:", error);

    // Try cache as fallback
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }

    return new Response("Offline - API not available", { status: 503 });
  }
}

async function handleDefaultRequest(request) {
  try {
    const response = await fetch(request);
    return response;
  } catch (error) {
    console.error("Default request failed:", error);
    return new Response("Offline", { status: 503 });
  }
}

// Background sync for offline actions
self.addEventListener("sync", (event) => {
  if (event.tag === "background-sync") {
    event.waitUntil(doBackgroundSync());
  }
});

async function doBackgroundSync() {
  console.log("Performing background sync...");
  // Implement background sync logic here
}

// Push notifications (if needed)
self.addEventListener("push", (event) => {
  const options = {
    body: event.data ? event.data.text() : "New content available!",
    icon: "/assets/icon.png",
    badge: "/assets/badge.png",
    vibrate: [100, 50, 100],
    data: {
      dateOfArrival: Date.now(),
      primaryKey: 1,
    },
  };

  event.waitUntil(
    self.registration.showNotification("Junk is Amitco", options)
  );
});

// Notification click
self.addEventListener("notificationclick", (event) => {
  event.notification.close();

  event.waitUntil(clients.openWindow("/"));
});
