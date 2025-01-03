import { createReadStream } from "fs"
import { Listener } from "../listener.js"
import { register as apply, seekToStart } from "../input-log.js"
import { VersionRegistry } from "../version.js"
import { createInterface } from "readline"
import { Replay } from "../replays.js"

const replaysStream = createReadStream("data/replays-2.jsonl")
const lines = createInterface({
  input: replaysStream,
  crlfDelay: Infinity
})

let cnt = 0

for await (const line of lines) {
  if (cnt > 150) {
    const { uploadtime, inputlog }: Replay = JSON.parse(line)
    const inputs = inputlog.split("\n")

    let [{ formatId, ...seed }, i] = seekToStart(inputs, 0)

    const listener = new Listener()

    function flush() {
      battle.sendUpdates()
      listener.flush()
    }

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
      flush()
      apply(battle, inputs[i])
      i += 1
    }
  }
  console.log(cnt)
  cnt += 1
}
