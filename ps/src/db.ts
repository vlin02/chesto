import { Db, ObjectId } from "mongodb"
import { Log } from "./replay.js"

export const DB_URL = "mongodb://localhost:27017"

export async function createReplays(db: Db, name: string) {
  await db.createCollection(name, {
    storageEngine: { wiredTiger: { configString: "block_compressor=zstd" } }
  })

  const col = db.collection(name)
  await col.createIndex({ rating: 1 })
  await col.createIndex({ uploadtime: 1 })
  await col.createIndex({ id: 1 })
  await col.createIndex({ format: 1 })

  return col
}

export type Replay = {
  _id: ObjectId
  id: string
  uploadtime: number
  players: [string, string]
  rating: number
  private: number
  password: string | null
  log: Log[][] | null
  inputlog: string
}
