import { parentPort } from "worker_threads"
import { Environment } from "./env.js"
import { Side, SIDES } from "../battle.js"
import { Action } from "../parser/action.js"

const envs = new Map<string, Environment>()

export type Request = [
  string,
  { type: "start" } | { type: "step"; actions: Action[] } | { type: "close" }
]

export type Update =
  | {
      done: true
      winner: Side | null
    }
  | {
      done: false
    }

const main = parentPort!

function step(env: Environment, actions: Action[]): Update {
  const event = env.step(actions)

  switch (event.type) {
    case "end":
      const { winner } = event
      return {
        done: true,
        winner
      }
    case "request": {
      return {
        done: false
      }
    }
  }
}

main!.on("message", ([id, body]: Request) => {
  switch (body.type) {
    case "start": {
      const fixed = [SIDES[Math.floor(Math.random() * 2)]]

      const env = new Environment(fixed)
      envs.set(id, env)

      main.postMessage([id, step(env, [])])
      break
    }
    case "step": {
      const { actions } = body
      const env = envs.get(id)!

      main.postMessage([id, step(env, actions)])
      break
    }
    case "close": {
      envs.delete(id)
      break
    }
  }
})
