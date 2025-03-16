import { Hono } from "hono"
import { Worker } from "worker_threads"
import { randomUUID } from "crypto"
import path, { dirname } from "path"
import { fileURLToPath } from "url"
import { EnvUpdate, Action } from "./env.js"
import { WorkerRequest } from "./worker.js"
import { Side } from "../battle.js"

const NUM_WORKERS = 60
const __filename = fileURLToPath(import.meta.url)
const WORKER_PATH = path.join(dirname(__filename), "./worker.js")

type Session = {
  workerId: number
  resolve: (update: EnvUpdate) => void
}

const sessions = new Map<string, Session>()

const workers = Array.from({ length: NUM_WORKERS }, () => new Worker(WORKER_PATH))

workers.forEach((worker) => {
  worker.on("message", ([id, update]: [string, EnvUpdate]) => {
    sessions.get(id)!.resolve!(update)
  })
})

function sendMessage(id: number, req: WorkerRequest) {
  workers[id].postMessage(req)
}

const app = new Hono()

app.post("/start", async (c) => {
  const envId = randomUUID()
  const workerId = Math.floor(Math.random() * NUM_WORKERS)
  const { auto } = await c.req.json<{ auto: Side[] }>()

  const update = await new Promise<EnvUpdate>((resolve) => {
    sessions.set(envId, { workerId, resolve })
    sendMessage(workerId, [envId, { type: "start", auto }])
  })

  return c.json({ id: envId, update })
})

app.post("/:id/step", async (c) => {
  const envId = c.req.param("id")
  const actions = await c.req.json<Action[]>()

  const session = sessions.get(envId)!
  const { workerId } = session

  const update = await new Promise<EnvUpdate>((resolve) => {
    session.resolve = resolve
    sendMessage(workerId, [envId, { type: "step", actions }])
  })

  return c.json(update)
})

app.delete("/:id", async (c) => {
  const envId = c.req.param("id")
  const session = sessions.get(envId)!
  const { workerId } = session

  sendMessage(workerId, [envId, { type: "close" }])
  sessions.delete(envId)

  return c.json({})
})

export default app
