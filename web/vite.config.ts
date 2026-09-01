import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { VitePWA } from "vite-plugin-pwa";

export default defineConfig({
  plugins: [react(), tailwindcss(), VitePWA({
    registerType: "prompt",
    includeAssets: ["face-check.svg", "face-check-maskable.svg"],
    manifest: {
      name: "Face Check", short_name: "Face Check",
      description: "Private, explainable photo face verification.",
      theme_color: "#10231d", background_color: "#eef1eb",
      display: "standalone", start_url: "/", scope: "/",
      icons: [
        { src: "/face-check.svg", sizes: "any", type: "image/svg+xml", purpose: "any" },
        { src: "/face-check-maskable.svg", sizes: "any", type: "image/svg+xml", purpose: "maskable" },
      ],
    },
    workbox: { navigateFallback: "/index.html", runtimeCaching: [{
      urlPattern: /\/(?:api\/|readyz$)/,
      handler: "NetworkOnly",
    }] },
  })],
  resolve: {
    alias: {
      "@components": "/src/@components",
      "@hooks": "/src/@hooks",
      "@services": "/src/@services",
      "@interfaces": "/src/@interfaces",
      "@utils": "/src/@utils",
    },
  },
  server: {
    proxy: {
      // ws:true so the live camera stream's upgrade request is forwarded too;
      // without it /api/v1/face/stream never reaches the backend in development.
      "/api": { target: "http://localhost:8000", ws: true },
      "/healthz": "http://localhost:8000",
      "/readyz": "http://localhost:8000",
    },
  },
});
