import { MongoClient } from "mongodb"
import { DB_URL } from "./db.js"

const mongo = new MongoClient(DB_URL)
await mongo.connect()
const db = mongo.db("chesto")

console.log(await db.collection("replays").findOne({ id: "gen9randombattle-2003211807" }))

await mongo.close()
