import { Battle as Simulator, Teams, ID } from "@pkmn/sim"
import { Battle as Client } from "@pkmn/client"
import { TeamGenerators } from "@pkmn/randoms"
import { Generations } from "@pkmn/data"
import { Dex } from "@pkmn/dex"
import EventEmitter from "node:events"
import { Side, SIDES } from "./battle.js"

Teams.setGeneratorFactory(TeamGenerators)

export type Choice =
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

export type Request =
  | { retry: true }
  | {
      retry: false
      switch: boolean
    }

export type Event = ["request", { side: Side; req: Request }] | ["end"]

export type Action = {
  side: Side
  choice: Choice
}

class Player {
  state: Client

  constructor() {
    this.state = new Client(new Generations(Dex))
  }
}

export type ArenaEmitter = EventEmitter<{ event: [Event] }>

const SPLIT_REGEX = /^^\|split\|(.*)$/

export class Arena {
  sim: Simulator
  formatId: ID
  p1: Player
  p2: Player
  closed: boolean

  private chunks: [string, string | string[]][] = []

  emitter: ArenaEmitter

  constructor({ formatId }: { formatId: ID }) {
    this.formatId = formatId
    this.p1 = new Player()
    this.p2 = new Player()
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

          const [_, type, msg] = line.split("|")

          switch (type) {
            case "error": {
              if (
                ["Can't do anything:", "Can't undo:", "Can't make choices:"].some((x) =>
                  msg.startsWith(x)
                )
              ) {
                throw Error()
              }

              events.push([
                "request",
                {
                  side,
                  req: { retry: true }
                }
              ])

              break
            }
            case "request": {
              state.update()
              state.update()

              const { requestType } = state.request!

              if (requestType === "move" || requestType === "switch") {
                events.push([
                  "request",
                  {
                    side,
                    req: {
                      retry: false,
                      switch: requestType === "switch"
                    }
                  }
                ])
              }

              break
            }
            default:
              throw Error(line)
          }
          break
        }
        case "end": {
          events.push(["end"])
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
