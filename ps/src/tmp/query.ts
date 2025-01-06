import { MongoClient } from "mongodb"
import { DB_URL } from "../db.js"

const client = await new MongoClient(DB_URL).connect()
const db = client.db("chesto")

await db
  .collection("replays")
  .aggregate([{ $match: {} }, { $out: "replaysBackup" }])
  .next()

await client.close()
