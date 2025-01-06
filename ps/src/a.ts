import { MongoClient } from "mongodb"
import { DB_URL } from "./db.js"

const client = new MongoClient(DB_URL)
await client.connect()

const db = client.db("chesto")

const cursor = db.collection("replay").find({
  rating: { $exists: false }
})

// Get count
const count = await cursor.next()
console.log(count)
await client.close()