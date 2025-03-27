import { Observer } from "../parser/observer.js"
import { Choice, toMoves } from "../parser/option.js"

export function resolveChoice(obs: Observer, id: number): Choice {
  const opt = obs.getOption()!
  if (id < 8) {
    const j = id % 2
    const i = (id - j) / 2
    return {
      type: "move",
      move: toMoves(opt.select!)[i],
      tera: j === 1
    }
  }


  id -= 8
  return { type: "switch", species: [...Object.keys(obs.ally.team)][id] }
}
