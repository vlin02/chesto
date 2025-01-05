import { createReadStream, createWriteStream } from "fs"
import { deserialize, serialize } from "v8"
import { constants, createGunzip, createGzip } from "zlib"
import { BlockReader, CHUNK_SIZE, toUInt } from "../stream.js"
import { finished } from "stream/promises"
import { toLog } from "../client.js"

export function gunzip(path: string) {
  return createReadStream(path, { highWaterMark: CHUNK_SIZE }).pipe(
    createGunzip({
      chunkSize: CHUNK_SIZE,
      maxOutputLength: CHUNK_SIZE
    })
  )
}

export function gzip(path: string) {
  const gz = createGzip({
    level: 1,
    memLevel: 9,
    strategy: constants.Z_FILTERED,
    windowBits: 15,
    chunkSize: CHUNK_SIZE,
    maxOutputLength: CHUNK_SIZE
  })

  gz.pipe(createWriteStream(path, { highWaterMark: CHUNK_SIZE }))
  return gz
}

const reader = gunzip("data/replays.gz")
const writer = gzip("data/replays-2.gz")

const blockr = new BlockReader()

let i = 0
for await (const chunk of reader) {
  for (const block of blockr.load(chunk)) {
    const replay = deserialize(block)
    const { log } = replay

    const buf = serialize({
      ...replay,
      log: log.map(toLog)
    })
    writer.write(toUInt(buf.byteLength))
    writer.write(buf)

    if (++i % 1000 === 0) console.log(i)
  }
}

reader.close()
writer.end()
await finished(writer)
