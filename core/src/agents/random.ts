import { Observer } from "../parser/observer.js"
import { toChoices } from "../parser/option.js"

export function chooseRandom(obs: Observer) {
  const opt = obs.getOption()!
  const choices = toChoices(opt)

  const i = Math.floor(Math.random() * choices.length)

  return choices[i]
}
