import { ObjectId } from "mongodb"
import { apply, seekToStart } from "./protocol.js"
import { VersionManager } from "./version.js"

export type Log = ["update", string[]] | ["sideupdate", string] | ["end", string]

export type Replay = {
  _id: ObjectId
  id: string
  uploadtime: number
  players: [string, string]
  rating: number
  private: number
  password: string | null
  log: Log[]
  inputlog: string
}

export async function playback(uploadtime: number, inputs: string[]) {
  let [{ formatId, ...seed }, i] = seekToStart(inputs, 0)

  const vm = new VersionManager()
  const release = vm.getNearest(uploadtime)
  if (!release) throw Error()
  const { Battle } = await vm.set(release)

  let logs: Log[] = []

  const battle = new Battle({
    formatid: formatId,
    seed: seed.battle,
    p1: {
      seed: seed.p1
    },
    p2: {
      seed: seed.p2
    },
    send: (...log) => logs.push(log as Log)
  })

  while (i < inputs.length) {
    battle.sendUpdates()
    apply(battle, inputs[i])
    i += 1
  }

  return logs
}
