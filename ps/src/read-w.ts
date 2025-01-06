import { MongoClient, ObjectId } from "mongodb"
import { parentPort, workerData } from "worker_threads"
import { DB_URL } from "./db.js"

const i: number = workerData
const client = new MongoClient(DB_URL)
await client.connect()

const db = client.db("chesto")

const cursor = db.collection("replay").find({ uploadtime: { $mod: [7, i] } })

for await (const _ of cursor) {
  parentPort!.postMessage(0)
}

await client.close()
