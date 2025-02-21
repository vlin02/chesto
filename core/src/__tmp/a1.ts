import { MongoClient, ObjectId } from "mongodb"
import { DB_URL } from "./db.js"

const mongo = new MongoClient(DB_URL)
await mongo.connect()
const db = mongo.db("chesto")

let j = 0
for await (const replay of db.collection("replays").find({}, { projection: { steps: 1 } })) {
  let i = 0
  for (const s of replay["steps"]) {
    if (s !== null) i += 1
  }

  await db.collection("replays").updateOne({ _id: replay["_id"] }, { $set: { stepCount: i } })

  if (++j % 1000 === 0) {
    console.log(j)
  }
}
