import { resolve } from "path";
import { Worker } from "worker_threads";

const w = new Worker(resolve(import.meta.dirname, "b1.js"))
w.postMessage(["hello"])
