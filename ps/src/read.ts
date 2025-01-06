import { Worker } from "worker_threads"
import { resolve } from "path"

let k = 0
for (let i = 0; i < 7; i++) {
  const w = new Worker(resolve(import.meta.dirname, "read-w.js"), { workerData: i })

  w.on("message", (i) => {
    if (++k % 1000 === 0) {
      console.log(k)
    }
  })

  w.on("error", (e) =>{
    throw e
  })
}
