import { parentPort } from "worker_threads"
import { Replay } from "./replays.js"
import { apply, seekToStart } from "./input-log.js"
import { VersionRegistry } from "./version.js"

parentPort!.on("message", async (line: string) => {
  const replay: Replay = JSON.parse(line)
  const { uploadtime, inputlog } = replay

  const inputs = inputlog.split("\n")
  const log: any[] = []

  let [{ formatId, ...seed }, i] = seekToStart(inputs, 0)

  const registry = new VersionRegistry()
  const Battle = await registry.setByUnixSeconds(uploadtime)
  if (!Battle) throw Error()

  const battle = new Battle({
    formatid: formatId,
    seed: seed.battle,
    p1: {
      name: "p1",
      seed: seed.p1
    },
    p2: {
      name: "p2",
      seed: seed.p2
    },
    send: (...x) => log.push(x)
  })

  while (i < inputs.length) {
    apply(battle, inputs[i])
    i += 1
    battle.sendUpdates()
  }

  parentPort!.postMessage(JSON.stringify({ ...replay, log }))
})
