import { Battle, Teams, ID, PRNGSeed, PRNG } from "@pkmn/sim"
import { Battle as Client } from "@pkmn/client"
import { TeamGenerators } from "@pkmn/randoms"
import { Generations } from "@pkmn/data"
import { Dex } from "@pkmn/dex"
import { Side, SIDES } from "./battle.js"
import { compare } from "./util.js"

Teams.setGeneratorFactory(TeamGenerators)
export type Event = ["choice", { side: Side; error: boolean }] | ["turn"] | ["end"]

class Player {
  client: Client

  constructor() {
    this.client = new Client(new Generations(Dex))
  }
}

const SPLIT_REGEX = /^^\|split\|(.*)$/

export class Listener {
  p1: Player
  p2: Player
  closed: boolean
  receive: (type: string, data: string | string[]) => void
  private chunks: [string, string | string[]][] = []

  constructor() {
    this.p1 = new Player()
    this.p2 = new Player()
    this.chunks = []
    this.closed = false

    this.receive = (type, data) => {
      this.chunks.push([type, data])
    }
  }

  flush() {
    let events: Event[] = []
    let i = 0
    while (i < this.chunks.length) {
      const [type, data] = this.chunks[i]

      let lines = Array.isArray(data) ? data : [data]

      switch (type) {
        case "update": {
          let j = 0

          while (j < lines.length) {
            if (lines[j].startsWith("|turn")) {
              events.push(["turn"])
            }

            let match = lines[j].match(SPLIT_REGEX)
            if (match) {
              const secretSide = match[1]
              const secret = lines[j + 1]
              const shared = lines[j + 2]

              for (const side of SIDES) {
                this[side].client.add(secretSide === side ? secret : shared)
              }

              j += 3
              continue
            }

            for (const side of SIDES) {
              this[side].client.add(lines[j])
            }

            j += 1
          }
          break
        }
        case "sideupdate": {
          const [side, line] = lines[0].split("\n") as [Side, string]

          const agent = this[side]
          const { client } = agent
          client.add(line)

          const [_, type] = line.split("|")

          switch (type) {
            case "error": {
              events.push([
                "choice",
                {
                  side,
                  error: true
                }
              ])

              break
            }
            case "request": {
              client.update()
              client.update()

              const { requestType } = client.request!

              if (requestType === "move" || requestType === "switch") {
                events.push([
                  "choice",
                  {
                    side,
                    error: false
                  }
                ])
              } else if (requestType === "team") {
                throw Error()
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
    return events.sort(
      compare(([type]) => {
        return [{ end: 0, turn: 1, choice: 2 }[type]]
      })
    )
  }
}
