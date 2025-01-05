import { deserialize } from "v8"
import { BlockReader, CHUNK_SIZE } from "../stream.js"
import { createReadStream } from "fs"
import { createGunzip } from "zlib"
import { Client } from "../client.js"

export function gunzip(path: string) {
  return createReadStream(path, { highWaterMark: CHUNK_SIZE }).pipe(
    createGunzip({
      chunkSize: CHUNK_SIZE,
      maxOutputLength: CHUNK_SIZE
    })
  )
}

const reader = gunzip("data/replays-2.gz")

const blockr = new BlockReader()

let i = 0
for await (const chunk of reader) {
  for (const block of blockr.load(chunk)) {
    const replay = deserialize(block)
    const client = new Client()
    const { log } = replay
    
    client.consume(log)

    if (++i % 1000 === 0) console.log(i)
  }
}

reader.close()
