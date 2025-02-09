import { FOE, Side, SIDES } from "./client/protocol.js"
import { piped, spaced } from "./parse.js"

export type Log = ["update", string[]] | ["sideupdate", string] | ["end", string]

export function split(log: Log) {
  let p

  const chs: { p1: string[]; p2: string[] } = { p1: [], p2: [] }

  const [type] = log

  switch (type) {
    case "update": {
      const [, lines] = log

      let i = 0
      while (i < lines.length) {
        const line = lines[i]

        p = piped(line, 1)

        if (p.args[0] === "split") {
          p = piped(line, p.i)
          const side = p.args[0] as Side

          chs[side].push(lines[i + 1])
          chs[FOE[side]].push(lines[i + 2])

          i += 3
        } else {
          for (const side of SIDES) chs[side].push(line)
          i += 1
        }
      }
      break
    }
    case "sideupdate": {
      const [, line] = log
      const side = line.slice(0, 2) as Side
      const msg = line.slice(3)

      chs[side].push(msg)
      break
    }
    case "end": {
      const [, line] = log
      const msg = `|end|${line}`
      for (const side of SIDES) chs[side].push(msg)
      break
    }
  }

  return chs
}

type Choice =
  | {
      type: "move"
      name: string
      tera: boolean
    }
  | {
      type: "switch"
      i: number
    }

type Input =
  | {
      type: "version"
      origin: boolean
      hash: string
    }
  | {
      type: "start"
      seed: number[]
    }
  | {
      type: "choose"
      side: Side
      choice: Choice
    }
  | {
      type: "end"
      winner: Side | null
    }
  | { type: "chat" | "player" }

export function parseInput(line: string): Input {
  let p = spaced(line, 0)
  const pfx = p.args[0]

  switch (pfx) {
    case ">version":
    case ">version-origin": {
      const origin = pfx === ">version-origin"
      p = spaced(line, p.i)
      return { type: "version", origin, hash: pfx }
    }
    case ">player":
      return { type: "player" }
    case ">start": {
      const { seed } = JSON.parse(line.slice(p.i))
      return { type: "start", seed }
    }
    case ">p1":
    case ">p2": {
      p = spaced(line, p.i, -1)

      let choice: Choice
      switch (p.args[0]) {
        case "move":
          choice = { type: "move", name: p.args[1], tera: p.args[2] === "terastallize" }
          break
        case "switch":
          choice = { type: "switch", i: Number(p.args[1]) }
          break
        default:
          throw Error()
      }

      return { type: "choose", side: pfx.slice(1) as Side, choice }
    }
    case ">forcelose": {
      p = spaced(line, p.i)
      return { type: "end", winner: p.args[0] as Side }
    }
    case ">forcetie": {
      return { type: "end", winner: null }
    }
    case ">chat":
      return { type: "chat" }
    default:
      throw Error(pfx)
  }
}
