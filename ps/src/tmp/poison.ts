import { readFileSync } from "fs"
import { Listener } from "../listener.js"
import { seekToStart, register } from "../input-log.js"
import { Battle } from "@pkmn/sim"

const { inputlog } = JSON.parse(readFileSync("/Users/vilin/chesto/ps/data/poison.json", "utf-8"))
const inputs: string[] = inputlog.split("\n")

let [{ formatId, rated, ...seed }, i] = seekToStart(inputs, 0)

const listener = new Listener()

function flush() {
  for (const [type, v] of listener.flush()) {
    if (type === "turn") {
      if (battle.turn === 5) {
        console.log("here")
      }
    }
  }
}

const battle = new Battle({
  formatid: formatId,
  seed: seed.battle,
  rated,
  send: listener.receive
})
battle.setPlayer("p1", {
  name: "p1",
  seed: seed.p1
})
battle.setPlayer("p2", {
  name: "p2",
  seed: seed.p2
})

console.log(i, inputs)

flush()
