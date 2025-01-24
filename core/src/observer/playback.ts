import { MongoClient } from "mongodb"
import { parentPort, workerData } from "worker_threads"
//@ts-ignore
import Sim from "sim"
import { Log } from "../log.js"
import { seekToStart, Replay, apply } from "../replay.js"

async function playback(inputs: string[]) {
  let i = 0

  const [{ formatId, seed }, j] = seekToStart(inputs, i)
  i = j

  let outputs: Log[][] = []
  let buf: Log[] = []

  const battle = new Sim.Battle({
    formatid: formatId,
    seed: seed.battle,
    p1: {
      seed: seed.p1
    },
    p2: {
      seed: seed.p2
    },
    send: (...log: any) => {
      const [type, v] = log as Log
      if (type === "sideupdate" && v.startsWith("|error|", 3)) throw Error("input error")

      buf.push(log as Log)
    }
  })
  const { p1, p2 } = battle

  function flush() {
    battle.sendUpdates()
    outputs.push(buf)
    buf = []
  }

  flush()

  for (; i < inputs.length; i++) {
    apply(battle, inputs[i])
    flush()
  }

  return { outputs, p1: p1.team, p2: p2.team }
}

const { workerId, version } = workerData as { workerId: number; version: string }

const client = new MongoClient("mongodb://localhost:27017")
await client.connect()

const db = client.db("chesto")
const Replays = db.collection<Replay>("replays")

let j = 0
for await (const { _id, inputs } of Replays.find(
  { uploadtime: { $mod: [7, workerId] }, version },
  { projection: { inputs: 1 } }
)) {
  const { outputs, p1, p2 } = await playback(inputs)
  await Replays.updateOne({ _id }, { $set: { outputsV2: outputs, teams: [p1, p2] } })
}

await client.close()

parentPort!.postMessage({})
