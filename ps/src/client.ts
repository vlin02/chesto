import { Battle } from "@pkmn/client"
import { Generations } from "@pkmn/data"
import { Dex } from "@pkmn/dex"
import { Side, SIDES } from "./protocol.js"
import { compare } from "./util.js"
import { Log } from "./replay.js"

export type Event = ["choice", { side: Side; retry: boolean }] | ["turn"] | ["end"]

export function toLog([type, v]: ["update", string[]] | ["sideupdate", string] | ["end", string]): Log {
  switch (type) {
    case "update": {
      return [type, v]
    }
    case "sideupdate": {
      const side = v.slice(0, 2) as Side
      let j = v.indexOf("|", 4)
      if (v.startsWith("error", 4)) {
        return [
          type,
          {
            side,
            type: "error",
            message: v.slice(j + 1)
          }
        ]
      } else {
        return [type, { side, type: "request", request: JSON.parse(v.slice(j + 1)) }]
      }
    }
    case "end": {
      return [type, JSON.parse(v)]
    }
  }
}

export class Client {
  p1: Battle
  p2: Battle

  constructor() {
    this.p1 = new Battle(new Generations(Dex))
    this.p2 = new Battle(new Generations(Dex))
  }

  consume(logs: Log[]) {
    let events: Event[] = []

    for (const [type, v] of logs) {
      switch (type) {
        case "update": {
          let j = 0
          const lines = v

          while (j < lines.length) {
            const line = lines[j]

            if (line.startsWith("turn", 1)) {
              events.push(["turn"])
            }

            if (line.startsWith("split", 1)) {
              const secretSide = line.slice(-2)
              const secret = lines[j + 1]
              const shared = lines[j + 2]

              for (const side of SIDES) {
                this[side].add(secretSide === side ? secret : shared)
              }

              j += 3
              continue
            }

            for (const side of SIDES) {
              this[side].add(lines[j])
            }

            j += 1
          }
          break
        }
        case "sideupdate": {
          const { side } = v

          switch (v.type) {
            case "error": {
              events.push([
                "choice",
                {
                  side,
                  retry: true
                }
              ])

              break
            }
            case "request": {
              const p = this[side]
              const { request } = v
              console.log(request)
              if (request.wait)

              p.request = request
              p.requestStatus = "applicable"
              p.update()

              events.push([
                "choice",
                {
                  side,
                  retry: false
                }
              ])
            }
          }

          break
        }
        case "end": {
          events.push(["end"])
          break
        }
      }
    }

    return events.sort(
      compare(([type]) => {
        return [{ end: 0, turn: 1, choice: 2 }[type]]
      })
    )
  }
}
