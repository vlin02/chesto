import { Db, MongoClient } from "mongodb"

export const DB_URL = "mongodb://localhost:27017"

async function createReplay(db: Db, name: string) {
  await db.createCollection(name)

  const col = db.collection(name)
  await col.createIndex({ rating: 1 })
  await col.createIndex({ uploadtime: 1 })
  await col.createIndex({ id: 1 })
  await col.createIndex({ format: 1 })
}

