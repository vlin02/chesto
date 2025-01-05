import { createReadStream } from "fs"
import { createInterface } from "readline"
import { partition } from "./util.js"
import { Worker } from "worker_threads"
import { resolve } from "path"

const replaysStream = createReadStream("data/replays-2.jsonl")
const rl = createInterface({
  input: replaysStream,
  crlfDelay: Infinity
})
const lines = []
for await (const line of rl) {
  lines.push(line)
}

const CORES = 7

const chunks = partition(lines.slice(0, 1000), Math.ceil(lines.length / CORES))

let cnt = 0

for (const chunk of chunks) {
  const worker = new Worker(resolve(import.meta.dirname, "worker.js"), {
    workerData: chunk
  })

  worker.on("message", ([type, id]: [string, string]) => {
    if (type === "done") {
      cnt += 1
      if (cnt % 100 === 0) {
        console.log(cnt)
      }
    } else {
      throw Error(id)
    }
  })
}
