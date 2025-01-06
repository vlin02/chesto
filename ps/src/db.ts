import { Db } from "mongodb"

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
