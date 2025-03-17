import { serve } from "@hono/node-server"
import app from "./server.js"
const PORT = 3000

serve({
  fetch: app.fetch,
  port: PORT,
  serverOptions: {
    keepAliveTimeout: 0, // Disable keep-alive timeout (connections stay open indefinitely)
  }
})

console.log(`Worker pool server running at http://localhost:${PORT}`)
