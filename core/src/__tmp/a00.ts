import { MongoClient } from "mongodb"
import { workerData } from "worker_threads"
import { Replay } from "../db.js"
import { DB_URL } from "./db.js"

const { i, count } = workerData

const mongo = new MongoClient(DB_URL)
await mongo.connect()
const db = mongo.db("chesto")

let j = 0

for await (const { _id, id } of db
  .collection<Replay>("replays")
  .find({}, { projection: { id: 1 } })) {
  await db
    .collection<Replay>("replays")
    .updateOne({ _id }, { $set: { num: Number(id.split("-")[1]) } })

  if (++j % 100 === 0) console.log(j)
}

await mongo.close()
