import { Battle } from "@pkmn/client"
import { Generations } from "@pkmn/data"
import { Dex } from "@pkmn/dex"
import { Side, SIDES } from "./battle.js"
import { compare } from "./util.js"
import { Log } from "./replay.js"

export type Event = ["choice", { side: Side; retry: boolean }] | ["turn"] | ["end"]

export class Observer {
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
          const line = v as string

          const side = line.slice(0, 2) as Side
          const p = this[side]

          p.add(line.slice(3))

          if (line.startsWith("request", 4)) {
            p.update()
            p.update()

            events.push([
              "choice",
              {
                side,
                retry: false
              }
            ])
          } else {
            events.push([
              "choice",
              {
                side,
                retry: true
              }
            ])
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
