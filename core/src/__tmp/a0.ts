import { MongoClient } from "mongodb"
import { Observer } from "../client/observer.js"
import { Generations } from "@pkmn/data"
import { Dex } from "@pkmn/dex"
import { parseInput, split } from "../log.js"
import { SIDES } from "../client/protocol.js"
import { workerData } from "worker_threads"
import { VersionCache, Step, Replay } from "../db.js"
import { Format, Run, toChoice } from "../run.js"
import { extractObservation } from "../features/observation.js"
import { extractOptions } from "../features/options.js"
import { DB_URL } from "./db.js"

// const { i, count } = workerData

const mongo = new MongoClient(DB_URL)
await mongo.connect()
const db = mongo.db("chesto")

const vc = new VersionCache(db)

let j = 0
const gen = new Generations(Dex).get(9)

for await (const { _id, inputs, outputs, version } of db.collection<Replay>("replays").find(
  {},
  { projection: { inputs: 1, outputs: 1, version: 1 } }
).limit(2000)) {
  const obs = { p1: new Observer(gen), p2: new Observer(gen) }

  const { patch } = await vc.load(version)
  const fmt: Format = { gen, patch }

  const steps: (Step | null)[] = []

  for (let i = 0; i < inputs.length; i++) {
    const j = i + outputs.length - inputs.length

    const line = inputs[i]
    const input = parseInput(line)

    const logs = j < 0 ? [] : outputs[j]

    let step: Step | null = null

    if (input.type === "choose") {
      const { side } = input
      const run: Run = { fmt, obs: obs[side] }

      step = {
        side,
        observation: extractObservation(fmt, obs[side]),
        options: extractOptions(run),
        choice: toChoice(run, input.choice)
      }
    }

    for (const log of logs) {
      const ch = split(log)
      for (const side of SIDES) {
        for (const msg of ch[side]) {
          obs[side].read(msg)
        }
      }
    }

    steps.push(step)
  }

  await db.collection("replays").updateOne(
    { _id },
    {
      $set: {
        steps
      }
    }
  )

  if (++j % 100 === 0) console.log(j)
}

await mongo.close()
