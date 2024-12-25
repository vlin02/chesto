import { toID } from "@pkmn/sim"
import { Arena, ArenaEvent, Choice } from "./arena.js"
import { Side } from "./battle.js"

type Request =
  | {
      retry: true
    }
  | {
      retry: false
      forceSwitch: boolean
    }

interface Agent {
  side: Side
  step: (req: Request) => Choice | Promise<Choice>
}

class Handler {
  arena: Arena
  agent: Agent

  constructor(arena: Arena, agent: Agent) {
    this.arena = arena
    this.agent = agent
  }

  async handle(e: ArenaEvent) {
    if (this.agent.side === e.side) {
      let choice: Choice

      if (e.type === "request") {
        choice = await this.agent.step({
          retry: false,
          forceSwitch: e.forceSwitch
        })
      } else if (e.type === "retry") {
        choice = await this.agent.step({
          retry: true
        })
      }

      this.arena.send({
        side: this.agent.side,
        choice: choice!
      })
    }
  }
}

const arena = new Arena({
  formatId: toID("gen9randombattle")
})

class AutoAgent implements Agent {
  side: Side

  constructor(side: Side) {
    this.side = side
  }

  step(): Choice {
    return { type: "auto" }
  }
}

const p1 = new Handler(arena, new AutoAgent("p1"))
const p2 = new Handler(arena, new AutoAgent("p2"))

arena.emitter.on("event", (e) => {
  p1.handle(e)
  p2.handle(e)
})

arena.start()
