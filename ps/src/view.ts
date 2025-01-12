import { Battle } from "@pkmn/client"
import { Generations, TypeName } from "@pkmn/data"
import { Dex } from "@pkmn/dex"
import { Side, SIDES } from "./sim.js"
import { compare } from "./util.js"
import { Log } from "./replay.js"

export type Event = ["choice", { side: Side; retry: boolean }] | ["turn"] | ["end"]

function parsePokemonId(s: string) {
  const side = s.slice(0, 2) as Side
  const speciesId = s.slice(5)
  return [side, speciesId] as const
}

export class View {
  side: Side

  p1: {
    terastallized?: [string, TypeName]
    revealed: Set<string>
    active?: string
  }

  p2: {
    terastallized?: [string, TypeName]
    revealed: Set<string>
    active?: string
  }

  constructor() {
    this.p1 = { revealed: new Set() }
    this.p2 = { revealed: new Set() }
  }

  add(line: string) {
    const parts = line.split("|")
    if (parts[1] === "-terastallize") {
      const [side, speciesId] = parsePokemonId(parts[2])
      this[side].terastallized = [speciesId, parts[3] as TypeName]
    }

    if (parts[1] === "switch" || parts[1] === "drag") {
      const [side, speciesId] = parsePokemonId(parts[2])
      this[side].revealed.add(speciesId)
      this[side].active = speciesId
    }

    if (parts[1] === "replace") {
      const [side, speciesId] = parsePokemonId(parts[2])
      const s = this[side]
      s.revealed.delete(s.active!)
      s.active = speciesId
      s.revealed.add(speciesId)
    }
  }
}

export class Observer {
  p1: {
    b: Battle
    v: View
  }
  p2: {
    b: Battle
    v: View
  }

  constructor() {
    this.p1 = { b: new Battle(new Generations(Dex)), v: new View() }
    this.p2 = { b: new Battle(new Generations(Dex)), v: new View() }
  }

  private add(side: Side, line: string) {
    this[side].b.add(line)
    this[side].v.add(line)
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
                this.add(side, secretSide === side ? secret : shared)
              }

              j += 3
              continue
            }

            for (const side of SIDES) {
              this.add(side, lines[j])
            }

            j += 1
          }
          break
        }
        case "sideupdate": {
          const line = v as string

          const side = line.slice(0, 2) as Side
          const { b } = this[side]

          this.add(side, line.slice(3))

          if (line.startsWith("request", 4)) {
            b.update()
            b.update()

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
