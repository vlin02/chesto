import { createReadStream } from "fs"
import { MongoClient } from "mongodb"
import { deserialize } from "v8"
import { createGunzip } from "zlib"
import { CHUNK_SIZE, BlockReader } from "./stream.js"
import { createReplays } from "./db.js"

const client = new MongoClient("mongodb://localhost:27017")
await client.connect()
const db = client.db("chesto")
const replay = await createReplays(db, "replays-3")

export function gunzip(path: string) {
  return createReadStream(path, { highWaterMark: CHUNK_SIZE }).pipe(
    createGunzip({
      chunkSize: CHUNK_SIZE,
      maxOutputLength: CHUNK_SIZE
    })
  )
}

const reader = gunzip("data/replays.gz")

const blockr = new BlockReader()

let buffer = []
const inserts = []
let i = 0
let inflight = 0
for await (const chunk of reader) {
  if (inflight > 1000) reader.pause()
  else reader.resume()
  for (const block of blockr.load(chunk)) {
    const x = deserialize(block)
    buffer.push(x)
    if (buffer.length === 1000) {
      inflight += 1000
      inserts.push(replay.insertMany(buffer).then(() => (inflight -= 1000)))
      buffer = []
    }
    if (++i % 1000 === 0) {
      console.log(i)
    }
  }
}

if (buffer.length) {
  inserts.push(replay.insertMany(buffer))
}

await Promise.all(inserts)
await client.close()
