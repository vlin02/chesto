import { Hono } from "hono"
import { Worker } from "worker_threads"
import { randomUUID } from "crypto"
import path, { dirname } from "path"
import { fileURLToPath } from "url"
import { Update, Action } from "./env.js"
import { WorkerRequest } from "./worker.js"
import { Side } from "../battle.js"
import { BSON } from "mongodb"

const NUM_WORKERS = 60
const __filename = fileURLToPath(import.meta.url)
const WORKER_PATH = path.join(dirname(__filename), "./worker.js")

type Session = {
  workerId: number
  resolve: (update: Update) => void
}

const sessions = new Map<string, Session>()

const workers = Array.from({ length: NUM_WORKERS }, () => new Worker(WORKER_PATH))
let i = 0

workers.forEach((worker) => {
  worker.on("message", ([id, update]: [string, Update]) => {
    sessions.get(id)!.resolve!(update)
  })
})

function sendMessage(id: number, req: WorkerRequest) {
  workers[id].postMessage(req)
}

const app = new Hono()

app.post("/start", async (c) => {
  const autos = await c.req.json<Side[][]>()

  return new Response(
    BSON.serialize({
      results: await Promise.all(
        autos.map(async (auto) => {
          const workerId = i
          i = (i + 1) % NUM_WORKERS

          const envId = randomUUID()
          const update = await new Promise<Update>((resolve) => {
            sessions.set(envId, { workerId, resolve })
            sendMessage(workerId, [envId, { type: "start", auto }])
          })
          return { id: envId, update }
        })
      )
    })
  )
})

app.post("/step", async (c) => {
  const reqs = await c.req.json<{ id: string; action: Action }[]>()

  const x = BSON.serialize({
    results: await Promise.all(
      reqs.map(async ({ id: envId, action }) => {
        const session = sessions.get(envId)!
        const { workerId } = session

        const update = await new Promise<Update>((resolve) => {
          session.resolve = resolve
          sendMessage(workerId, [envId, { type: "step", action }])
        })

        return update
      })
    )
  })

  return new Response(x)
})

app.post("/close", async (c) => {
  const ids = await c.req.json<string[]>()

  ids.map((envId) => {
    const session = sessions.get(envId)!
    const { workerId } = session

    sendMessage(workerId, [envId, { type: "close" }])
    sessions.delete(envId)
  })

  return c.json({})
})

app.get("/count", async (c) => {
  return c.json({
    count: sessions.size
  })
})

export default app
