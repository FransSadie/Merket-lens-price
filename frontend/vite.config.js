import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      "/health": "http://127.0.0.1:8000",
      "/ingest": "http://127.0.0.1:8000",
      "/pipeline": "http://127.0.0.1:8000",
      "/model": "http://127.0.0.1:8000",
      "/predict": "http://127.0.0.1:8000",
      "/data": "http://127.0.0.1:8000",
      "/run": "http://127.0.0.1:8000",
      "/docs": "http://127.0.0.1:8000",
      "/prediction": "http://127.0.0.1:8000"
    }
  }
});
