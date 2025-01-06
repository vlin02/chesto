import { Observer } from "../client.js"
import { deserialize } from "v8"
import { createWorker } from "./dist.js"

createWorker((block) => {
  const replay = deserialize(block)
  const client = new Observer()
  const { log } = replay

  client.consume(log)

  return client.p1.turn
})
