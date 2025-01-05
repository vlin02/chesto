import { toID, ID, PRNGSeed } from "@pkmn/sim"
import { AnyBattle } from "./version.js"

export type Side = "p1" | "p2"
export const SIDES = ["p1", "p2"] as const

export type Decision =
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

export function make(x: Decision) {
  switch (x.type) {
    case "switch": {
      const { i } = x
      return `switch ${i + 1}`
    }
    case "move": {
      const { i, event } = x
      let cmd = `move ${i + 1}`
      return event ? `${cmd} ${event}` : cmd
    }
    case "auto": {
      return "default"
    }
  }
}

export function seekToStart(lines: string[], i: number) {
  let mark = { start: false, p1: false, p2: false }
  let state: any = {}

  for (; i < lines.length; i++) {
    const line = lines[i]
    let j = line.indexOf(" ")
    const type = line.slice(0, j)
    switch (type) {
      case ">start": {
        const { formatid, seed, rated } = JSON.parse(line.slice(j + 1))
        state.battle = seed
        state.formatId = toID(formatid)
        state.rated = rated
        mark.start = true
        break
      }
      case ">player": {
        let k = line.indexOf(" ", j + 1)
        const side = line.slice(j + 1, k) as Side
        const { seed } = JSON.parse(line.slice(k + 1))

        state[side] = seed
        mark[side] = true
        break
      }
    }

    const { p1, p2, start } = mark
    if (start && p1 && p2) {
      return [
        state as {
          formatId: ID
          battle: PRNGSeed
          rated: boolean
          p1: PRNGSeed
          p2: PRNGSeed
        },
        i + 1
      ] as const
    }
  }

  throw Error()
}

export function apply(battle: AnyBattle, input: string) {
  let j = input.indexOf(" ")
  const type = input.slice(1, j === -1 ? undefined : j)

  switch (type) {
    case "p1":
    case "p2":
      battle.choose(type, input.slice(j + 1))
      break
    case "forcelose":
      let k = input.indexOf(" ", j + 1)
      const side = input.slice(j + 1, k) as Side
      battle.win(side)
      break
    case "forcetie":
      battle.win(null)
      break
    case "chat":
      break
    default:
      throw Error(type)
  }
}
