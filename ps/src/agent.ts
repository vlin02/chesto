// import { Listener, Event } from "./listener.js"
// import { Side } from "./battle.js"

// export interface Strategy {
//   step: (retry: boolean) => Decision | Promise<Decision>
// }

// export class Agent {
//   arena: Listener
//   strategy: Strategy
//   side: Side

//   constructor(arena: Listener, side: Side, agent: Strategy) {
//     this.arena = arena
//     this.strategy = agent
//     this.side = side
//   }

//   async handle([type, msg]: Event) {
//     const { strategy: agent, arena, side } = this

//     if (type === "choice") {
//       if (msg.side === side) {
//         const choice = await agent.step(msg.error)
//         arena.receive({ side, decision: choice })
//         return true
//       }
//     }

//     return false
//   }
// }

// export class Auto implements Strategy {
//   step(): Decision {
//     return { type: "auto" }
//   }
// }
