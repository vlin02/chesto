import { Worker } from "worker_threads"
import { resolve } from "path"

const count = 60

await Promise.all(
  [...Array(count).keys()].map(async (i) => {
    const w = new Worker(resolve(import.meta.dirname, "a00.js"), {
      workerData: { count, i },
      resourceLimits: {
        maxOldGenerationSizeMb: 4096
      }
    })

    await new Promise<void>((res, rej) => {
      w.on("message", (s) => {
        console.log(i, s)
      })

      w.on("exit", () => {
        res()
      })

      w.on("error", (e) => {
        console.log(i)
        rej(e)
      })
    })
  })
)
