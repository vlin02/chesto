import { readFileSync } from "fs"
import { Log, Observer, split } from "./log.js"
import { Battle, Teams } from "@pkmn/sim"
import { apply, seekToStart } from "./replay.js"
import { diffString } from "json-diff"
import { TeamGenerators } from "@pkmn/randoms"

function playback(inputs: string[]) {
  let i = 0
  let [{ formatId, seed }, j] = seekToStart(inputs, i)
  i = j

  let p1: string[] = []

  Teams.setGeneratorFactory(TeamGenerators)

  const battle = new Battle({
    formatid: formatId,
    seed: seed.battle,
    p1: {
      seed: seed.p1
    },
    p2: {
      seed: seed.p2
    },
    send: (...log) => {
      p1.push(...split(log as Log).p1)
    }
  })

  for (; i < inputs.length; i++) {
    apply(battle, inputs[i])
    battle.sendUpdates()
  }

  return p1
}

const id = "gen9randombattle-2278418129"
let res = await fetch(`https://replay.pokemonshowdown.com/${id}.inputlog`)
const inputs = (await res.text()).split("\n")

const obs = new Observer("p1")

const p1 = playback(inputs)
console.log(p1.join("\n"))
throw ""

let clone = {}
for (const log of p1) {
  obs.read(log)
  console.log(log)
  const tmp = JSON.parse(JSON.stringify(obs))
  console.log(diffString(clone, tmp))
  console.log("-----")
  clone = tmp
}
