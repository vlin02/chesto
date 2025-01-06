import { MongoClient } from "mongodb"
import { parentPort, workerData } from "worker_threads"
import { DB_URL } from "./db.js"
import { Observer } from "./observer.js"

const i: number = workerData
const client = new MongoClient(DB_URL)
await client.connect()

const db = client.db("chesto")

//@ts-ignore
const cursor = db.collection("replays-3").find({ uploadtime: { $mod: [7, i] } }, { log:1 })

for await (const { log } of cursor) {
  const obs = new Observer()
  obs.consume(log)
  parentPort!.postMessage(0)
}

await client.close()
