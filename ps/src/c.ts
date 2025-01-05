import { createReadStream, createWriteStream } from "fs"
import { resolve } from "path"
import { createInterface } from "readline"
import { Worker } from "worker_threads"
import { createGzip } from "zlib"

const rl = createInterface({
  input: createReadStream("data/replays-2.jsonl")
})

const writer = createWriteStream("data/out1.jsonl.gzip")
const gzip = createGzip()
gzip.pipe(writer)

const MAX_INFLIGHT = 100
let nSent = 0
let nFinished = 0
let done = false

const inflight = () => nSent - nFinished

const workers = [...Array(7)].map(() => {
  const w = new Worker(resolve(import.meta.dirname, "b.js"))

  w.on("message", (output) => {
    nFinished++
    if (inflight() <= MAX_INFLIGHT) rl.resume()

    gzip.write(output + "\n")
    if (nFinished % 100 === 0) console.log(nFinished)

    if (done && !inflight()) {
      workers.forEach((w) => w.terminate())
      gzip.end()
    }
  })

  return w
})

for await (const line of rl) {
  nSent++
  if (MAX_INFLIGHT < inflight()) rl.pause()
  workers[Math.floor(Math.random() * 7)].postMessage(line)
}

done = true
