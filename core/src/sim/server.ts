import { Hono } from "hono"
import { Worker } from "worker_threads"
import { randomUUID } from "crypto"
import { Action } from "@pkmn/sim"
import { Update } from "./worker.js"

const NUM_WORKERS = 4
const WORKER_PATH = "./worker.js"

const app = new Hono()

type Session = {
  workerId: number
  resolve: (update: Update) => void
}

const sessions = new Map<string, Session>()

const workers = Array.from({ length: NUM_WORKERS }, () => new Worker(WORKER_PATH))
workers.forEach((worker) => {
  worker.on("message", ([id, update]: [string, Update]) => {
    sessions.get(id)!.resolve!(update)
  })
})

app.post("/new", async (c) => {
  const id = randomUUID()
  const workerId = Math.floor(Math.random() * NUM_WORKERS)

  const update = await new Promise<Update>((resolve) => {
    sessions.set(id, { workerId, resolve })
    workers[workerId].postMessage([id, { type: "start" }])
  })

  return c.json({ id, update })
})

app.post("/:id/step", async (c) => {
  const id = c.req.param("id")
  const { actions } = await c.req.json<{ actions: Action[] }>()

  const session = sessions.get(id)!
  const { workerId } = session

  const update = await new Promise<Update>((resolve) => {
    session.resolve = resolve
    workers[workerId].postMessage([id, { type: "step", actions }])
  })

  return c.json({id, update})
})
