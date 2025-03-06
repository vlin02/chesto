import { toChoices } from "../parser/action.js"
import { Observer } from "../parser/observer.js"

export class RandomAgent {
  choose(obs: Observer) {
    const action = obs.getAction()

    const choices = toChoices(action)
    const i = Math.floor(Math.random() * choices.length)

    return choices[i]
  }
}
