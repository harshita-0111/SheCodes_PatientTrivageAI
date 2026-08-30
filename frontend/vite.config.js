import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// PatientTriage.ai — Vite configuration
// VITE_API_HOST lets Docker inject the service name "backend" so containers
// can reach each other. Falls back to "localhost" for plain `npm run dev`.
const apiHost = process.env.VITE_API_HOST || "localhost";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    watch: {
      usePolling: true,
    },
    proxy: {
      "/api": {
        target: `http://${apiHost}:8000`,
        changeOrigin: true,
      },
      "/ws": {
        target: `ws://${apiHost}:8000`,
        ws: true,
      },
    },
  },
  preview: {
    port: 4173,
  },
});
