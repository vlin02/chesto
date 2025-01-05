import { createReadStream } from "fs"
import { Client } from "../client.js"
import { VersionManager } from "../version.js"
import { createInterface } from "readline"
import { Replay } from "../replays.js"
import { apply, seekToStart } from "../protocol.js"


const replaysStream = createReadStream("data/replays-2.jsonl")
const lines = createInterface({
  input: replaysStream,
  crlfDelay: Infinity
})

let cnt = 0

for await (const line of lines) {
  if (7884 < cnt) {
    const { uploadtime, inputlog }: Replay = JSON.parse(line)
    const inputs = inputlog.split("\n")

    let [{ formatId, ...seed }, i] = seekToStart(inputs, 0)

    const listener = new Client()

    const registry = new VersionManager()
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
      listener.consume()

      apply(battle, inputs[i])
      i += 1
    }

    console.log(cnt)
  }

  cnt += 1
}
