import { MongoClient } from "mongodb"
import { DB_URL } from "./db.js"

const mongo = new MongoClient(DB_URL)
await mongo.connect()
const db = mongo.db("chesto")

const versionFrequency = await db.collection("replays").aggregate([
  { $group: { _id: "$version", count: { $sum: 1 } } },
  { $sort: { count: -1 } }
]).toArray();

console.log(versionFrequency)

await mongo.close()
