import { toChoices } from "../parser/option.js"
import { Observer } from "../parser/observer.js"

export class RandomAgent {
  obs: Observer

  constructor(obs: Observer) {
    this.obs = obs
  }
  choose() {
    const action = this.obs.getOption()

    const choices = toChoices(action)
    
    const i = Math.floor(Math.random() * choices.length)

    return choices[i]
  }
}
