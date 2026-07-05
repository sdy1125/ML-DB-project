import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_DEV_API_TARGET || 'http://localhost:8080',
        changeOrigin: true,
      },
      '/actuator': {
        target: process.env.VITE_DEV_API_TARGET || 'http://localhost:8080',
        changeOrigin: true,
      },
      '/ml': {
        target: process.env.VITE_DEV_ML_TARGET || 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/ml/, ''),
      },
    },
  },
})
