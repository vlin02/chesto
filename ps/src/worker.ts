import { parentPort } from "worker_threads"
import { Worker } from "worker_threads"

export function plog(...x: any[]) {
  parentPort!.postMessage(["log", ...x])
}

export function genericRun(path: string, cores: number) {
  for (let i = 0; i < cores; i++) {
    const w = new Worker(path, { workerData: i })

    w.on("message", ([type, ...rest]) => {
      if (type === "log") {
        console.log(i, ...rest)
      }
    })

    w.on("error", (e) => {
      throw e
    })
  }
}
