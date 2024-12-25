import { Request, Arena, Event, Choice } from "./arena.js"
import { Side } from "./battle.js"

interface Strategy {
  step: (req: Request) => Choice | Promise<Choice>
}

class Agent {
  arena: Arena
  strategy: Strategy
  side: Side

  constructor(arena: Arena, side: Side, agent: Strategy) {
    this.arena = arena
    this.strategy = agent
    this.side = side
  }

  async handle([type, msg]: Event) {
    const { strategy: agent, arena, side } = this

    if (type === "request") {
      const { req } = msg

      if (msg.side === side) {
        const choice = await agent.step(req)
        arena.send({ side, choice })
      }
    }
  }
}
