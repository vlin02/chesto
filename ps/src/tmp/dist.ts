import { parentPort, Worker } from "worker_threads"

export function createDistributed(f: string, cb: (output: any) => any) {
  let inflight = 0
  let done = false
  const workers = [...Array(7)].map(() => {
    const w = new Worker(f)

    return w
  })

  const p = new Promise<void>((resolve) => {
    for (const w of workers) {
      w.on("message", (output) => {
        inflight--
        cb(output)

        if (inflight === 0 && done) {
          workers.forEach((w) => w.terminate())
          resolve()
        }
      })
    }
  })

  return {
    inflight() {
      return inflight
    },
    async call(input: any) {
      inflight++
      workers[Math.floor(Math.random() * 7)].postMessage(input)
    },
    done() {
      done = true
      return p
    }
  }
}

export function createWorker(x: (a: any) => any) {
  parentPort!.on("message", (a: any) => {
    parentPort!.postMessage(x(a))
  })
}
