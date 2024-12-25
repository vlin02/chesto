import { Battle as Simulator, Teams, ID } from "@pkmn/sim"
import { Battle as Client } from "@pkmn/client"
import { TeamGenerators } from "@pkmn/randoms"
import { Generations } from "@pkmn/data"
import { Dex } from "@pkmn/dex"
import EventEmitter from "node:events"

Teams.setGeneratorFactory(TeamGenerators)

type Side = "p1" | "p2"

type Choice =
  | {
      type: "switch"
      i: 0 | 1 | 2 | 3 | 4 | 5
    }
  | {
      type: "move"
      i: 0 | 1 | 2 | 3
      event?: "zmove" | "ultra" | "mega" | "dynamax" | "terastallize"
    }
  | {
      type: "auto"
    }

function toCmd(choice: Choice) {
  switch (choice.type) {
    case "switch":
      return `switch ${choice.i + 1}`
    case "move":
      const { i, event } = choice
      let cmd = `move ${i + 1}`
      return event ? `${cmd} ${event}` : cmd
    case "auto":
      return "default"
  }
}

type Event =
  | {
      type: "retry"
      side: Side
    }
  | {
      type: "request"
      side: Side
      forceSwitch: boolean
    }
  | { type: "end" }

type Action = {
  side: Side
  choice: Choice
}

class Agent {
  state: Client

  constructor() {
    this.state = new Client(new Generations(Dex))
  }
}

const SIDES = ["p1", "p2"] as const
const SPLIT_REGEX = /^^\|split\|(.*)$/

type Emitter = EventEmitter<{ event: [Event] }>

export class Arena {
  sim: Simulator
  formatId: ID
  p1: Agent
  p2: Agent
  closed: boolean

  private chunks: [string, string | string[]][] = []

  emitter: Emitter

  constructor({ formatId }: { formatId: ID }) {
    this.formatId = formatId
    this.p1 = new Agent()
    this.p2 = new Agent()
    this.chunks = []
    this.emitter = new EventEmitter()
    this.closed = false

    this.sim = new Simulator({
      formatid: formatId,
      send: (type, data) => {
        this.chunks.push([type, data])
      }
    })
  }

  private flush() {
    this.sim.sendUpdates()

    let events: Event[] = []

    let i = 0
    while (i < this.chunks.length) {
      const [type, data] = this.chunks[i]

      let lines = Array.isArray(data) ? data : [data]

      switch (type) {
        case "update": {
          let j = 0

          while (j < lines.length) {
            let match = lines[j].match(SPLIT_REGEX)

            if (match) {
              const secretSide = match[1]
              const secret = lines[j + 1]
              const shared = lines[j + 2]

              for (const side of SIDES) {
                this[side].state.add(secretSide === side ? secret : shared)
              }

              j += 3
              continue
            }

            for (const side of SIDES) {
              this[side].state.add(lines[j])
            }

            j += 1
          }
          break
        }
        case "sideupdate": {
          const [side, line] = lines[0].split("\n") as [Side, string]

          const agent = this[side]
          const { state } = agent
          state.add(line)

          const [_, type, event] = line.split("|")

          switch (type) {
            case "error": {
              events.push({
                type: "retry",
                side
              })

              if (event.startsWith("[Unavailable choice]")) {
                i += 2
                continue
              }

              break
            }
            case "request": {
              state.update()
              state.update()

              const { requestType } = state.request!

              if (requestType === "move" || requestType === "switch") {
                events.push({
                  type: "request",
                  side,
                  forceSwitch: requestType === "switch"
                })
              }

              break
            }
            default:
              throw Error(line)
          }
          break
        }
        case "end": {
          events.push({ type: "end" })
          break
        }
        default:
          throw Error(type)
      }

      i += 1
    }

    this.chunks = []

    for (const event of events) {
      this.emitter.emit("event", event)
    }
  }

  send({ side, choice }: Action) {
    this.sim.choose(side, toCmd(choice))
    this.flush()
  }

  start() {
    this.sim.setPlayer("p1", {
      name: "bot1",
      team: Teams.generate(this.formatId)
    })
    this.sim.setPlayer("p2", {
      name: "bot2",
      team: Teams.generate(this.formatId)
    })

    this.flush()
  }

  close() {
    if (this.closed) return
    this.closed = true
    this.sim.destroy()
    this.emitter.removeAllListeners()
  }
}

export class EventStream {
  private buf: Event[]
  private resume?: (event: Event) => void
  private done: boolean

  constructor(emitter: Emitter) {
    this.buf = []
    this.done = false

    emitter.on("event", (e) => {
      this.buf.push(e)

      if (this.resume) {
        this.resume(this.buf.shift()!)
        this.resume = undefined
      }
    })
  }

  async next() {
    if (this.done) return ({ done: true } as const)

    const event =
      this.buf.shift() ??
      (await new Promise<Event>((resolve) => {
        this.resume = resolve
      }))

    if (event.type === "end") {
      this.done = true
    }

    return { done: false, value: event } as const
  }

  [Symbol.asyncIterator]() {
    return this
  }
}
