import { BlockReader, CHUNK_SIZE } from "../stream.js"
import { createReadStream, ReadStream } from "fs"
import { createGunzip } from "zlib"
import { resolve } from "path"
import { createDistributed } from "./dist.js"

const reader = createReadStream("data/replays.gz", { highWaterMark: CHUNK_SIZE })

const out = reader.pipe(
  createGunzip({
    chunkSize: CHUNK_SIZE,
    maxOutputLength: CHUNK_SIZE,
  })
)

const blockr = new BlockReader()

function check(reader: ReadStream, cnt: number) {
  if (cnt < 1000 && reader.isPaused()) reader.resume()
  if (cnt > 1000 && !reader.isPaused()) reader.pause()
}

function balance() {
  check(reader, dist.inflight())
}

let i = 0
const dist = createDistributed(resolve(import.meta.dirname, "r-w.js"), (t) => {
  if (++i % 1000 === 0) console.log(i, t)
  balance()
})

for await (const chunk of out) {
  for (const block of blockr.load(chunk)) {
    dist.call(block)
    balance()
  }
}
await dist.done()
reader.close()
