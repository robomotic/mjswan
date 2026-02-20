/*! coi-serviceworker v0.1.7 - Guido Zuidhof and contributors, licensed under MIT */
/*
 * Cross-Origin Isolation Service Worker
 *
 * On hosting platforms that don't allow custom HTTP headers (e.g. GitHub Pages),
 * this service worker intercepts all responses and injects the COOP/COEP headers
 * required for SharedArrayBuffer (used by MuJoCo WASM threading).
 */
if (typeof window === "undefined") {
  // Service Worker context
  self.addEventListener("install", () => self.skipWaiting());
  self.addEventListener("activate", (event) =>
    event.waitUntil(self.clients.claim())
  );
  self.addEventListener("fetch", (event) => {
    if (event.request.cache === "only-if-cached" && event.request.mode !== "same-origin") {
      return;
    }
    event.respondWith(
      fetch(event.request).then((response) => {
        if (response.status === 0) {
          return response;
        }
        const headers = new Headers(response.headers);
        headers.set("Cross-Origin-Embedder-Policy", "require-corp");
        headers.set("Cross-Origin-Opener-Policy", "same-origin");
        return new Response(response.body, {
          status: response.status,
          statusText: response.statusText,
          headers,
        });
      })
    );
  });
} else {
  // Window context – register the service worker
  (() => {
    const reloadedBySW = window.sessionStorage.getItem("coiReloadedBySW");
    window.sessionStorage.removeItem("coiReloadedBySW");

    // Already isolated, nothing to do
    if (window.crossOriginIsolated) return;

    const reg = navigator.serviceWorker;
    if (!reg) {
      console.warn("coi-serviceworker: navigator.serviceWorker is not available.");
      return;
    }

    reg.register(new URL("coi-serviceworker.js", window.location.href).href).then(
      (registration) => {
        if (registration.active && !registration.installing && !registration.waiting) {
          // SW is active but page is not isolated – need a reload
          if (!reloadedBySW) {
            window.sessionStorage.setItem("coiReloadedBySW", "true");
            window.location.reload();
          }
        }
        registration.addEventListener("updatefound", () => {
          const worker = registration.installing;
          if (!worker) return;
          worker.addEventListener("statechange", () => {
            if (worker.state === "activated" && !reloadedBySW) {
              window.sessionStorage.setItem("coiReloadedBySW", "true");
              window.location.reload();
            }
          });
        });
      },
      (err) => {
        console.error("coi-serviceworker: registration failed.", err);
      }
    );
  })();
}
