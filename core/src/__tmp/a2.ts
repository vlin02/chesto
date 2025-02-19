import { MongoClient } from "mongodb"
import { DB_URL } from "./db.js"

const mongo = new MongoClient(DB_URL)
await mongo.connect()
const db = mongo.db("chesto")

await db.collection("replays").updateMany({}, { $unset: { samples: 1 } })

await mongo.close()
