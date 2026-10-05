import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev server proxies the API and the WebSocket to the Python service so the
// dashboard can be developed with hot reload. `npm run build` emits to dist/,
// which the FastAPI app mounts and serves directly in production.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/ws': { target: 'ws://127.0.0.1:8000', ws: true },
    },
  },
  build: { outDir: 'dist', emptyOutDir: true },
})
