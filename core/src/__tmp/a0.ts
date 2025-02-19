import { MongoClient } from "mongodb"
import { Observer } from "../client/observer.js"
import { Generations } from "@pkmn/data"
import { Dex } from "@pkmn/dex"
import { parseInput, split } from "../log.js"
import { SIDES } from "../client/protocol.js"
import { workerData } from "worker_threads"
import { VersionCache, Step, withSchema } from "../db.js"
import { Format, Run, toChoice } from "../run.js"

const workerId = workerData as number

const mongo = new MongoClient("mongodb://172.31.30.235:27017")
await mongo.connect()

// const db = withSchema(mongo.db("chesto"))
// const vc = new VersionCache(db)

// let j = 0
// const gen = new Generations(Dex).get(9)
// for await (const { _id, inputs, outputs, version } of db.replays.find({
//   uploadtime: { $mod: [7, workerId] }
//   // id: "gen9randombattle-2003211807"
// })) {
//   const obs = { p1: new Observer(gen), p2: new Observer(gen) }

//   const { patch } = await vc.load(version)
//   const fmt: Format = { gen, patch }

//   const steps: Step[] = []

//   for (let i = 0; i < inputs.length; i++) {
//     const j = i + outputs.length - inputs.length

//     const line = inputs[i]
//     const input = parseInput(line)

//     const logs = j < 0 ? [] : outputs[j]

//     let sample: Sample | null = null
//     if (input.type === "choose") {
//       const { side } = input
//       const run: Run = { fmt, obs: obs[side] }

//       sample = {
//         observer: encodeObserver(fmt, obs[side]),
//         option: encodeOption(run),
//         choice: toChoice(run, input.choice)
//       }
//     }

//     for (const log of logs) {
//       const ch = split(log)
//       for (const side of SIDES) {
//         for (const msg of ch[side]) {
//           obs[side].read(msg)
//         }
//       }
//     }

//     const step = {
//       input: inputs[i],
//       logs,
//       sample
//     }

//     steps.push(step)
//   }

//   await db.replays.updateOne(
//     { _id },
//     {
//       $set: {
//         steps
//       }
//     }
//   )

//   if (++j % 1000 === 0) console.log(j)
// }
