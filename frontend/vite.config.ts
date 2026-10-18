import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Keep this project's Tailscale address available across frontend restarts.
// Additional exact hostnames can be supplied by the local operator.
const allowedHosts = [
  "annopilot-t003.tailb9a83c.ts.net",
  ...(process.env.ANNO_ALLOWED_HOSTS || "")
    .split(",")
    .map((hostname) => hostname.trim())
    .filter(Boolean),
];

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks: { charts: ["recharts"] },
      },
    },
  },
  server: {
    allowedHosts,
    port: 3000,
    strictPort: true,
    proxy: {
      "/api": {
        target: process.env.API_PROXY_TARGET || "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
