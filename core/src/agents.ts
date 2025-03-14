import { toChoices } from "./parser/option.js"
import { Observer } from "./parser/observer.js"

export class RandomAgent {
  obs: Observer

  constructor(obs: Observer) {
    this.obs = obs
  }

  choose() {
    const opt = this.obs.getOption()!
    const choices = toChoices(opt)

    const i = Math.floor(Math.random() * choices.length)

    return choices[i]
  }
}
