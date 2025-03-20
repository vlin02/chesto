import { toChoices } from "../parser/option.js"
import { Observer } from "../parser/observer.js"
import { Pool } from "undici"
import { encodeBattle, packBattle } from "../model/state.js"
import { toChoice } from "../env/env.js"

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

export class RLAgent {
  obs: Observer
  pool: Pool

  constructor(obs: Observer, pool: Pool) {
    this.obs = obs
    this.pool = pool
  }

  async choose() {
    let { body } = await this.pool.request({
      path: "/predict",
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify([packBattle(encodeBattle(this.obs))])
    })

    const ret = (await body.json()) as any
    const [{ action_id: actionId }] = ret

    return toChoice(this.obs, actionId)
  }
}
