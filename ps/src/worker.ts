import { parentPort, workerData } from "worker_threads"
import { Replay } from "./replays.js"
import { apply, seekToStart } from "./input-log.js"
import { Listener } from "./listener.js"
import { VersionRegistry } from "./version.js"

const lines: string[] = workerData

for (const line of lines) {
  const { id, uploadtime, inputlog }: Replay = JSON.parse(line)

  try {
    const inputs = inputlog.split("\n")

    let [{ formatId, ...seed }, i] = seekToStart(inputs, 0)

    const listener = new Listener()

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
      send: listener.receive
    })

    while (i < inputs.length) {
      battle.sendUpdates()
      listener.flush()

      apply(battle, inputs[i])
      i += 1
    }
    parentPort!.postMessage(["done", id])
  } catch (err) {
    parentPort!.postMessage(["error", id])
  }
}
