import { ID, PRNGSeed, toID } from "@pkmn/sim"
import { AnyBattle } from "./version.js"
import { FOE, piped, Side, SIDES } from "./parse/protocol.js"

export type Log = ["update", string[]] | ["sideupdate", string] | ["end", string]

type Header = {
  formatId: ID
  rated: boolean
  seed: {
    battle: PRNGSeed
    p1: PRNGSeed
    p2: PRNGSeed
  }
}

export function seekToStart(lines: string[], i: number) {
  let mark = { start: false, p1: false, p2: false }
  let header: any = { seed: {} }

  for (; i < lines.length; i++) {
    const line = lines[i]
    let j = line.indexOf(" ")
    const type = line.slice(0, j)
    switch (type) {
      case ">start": {
        const { formatid, seed, rated } = JSON.parse(line.slice(j + 1))
        header.seed.battle = seed
        header.formatId = toID(formatid)
        header.rated = rated
        mark.start = true
        break
      }
      case ">player": {
        let k = line.indexOf(" ", j + 1)
        const side = line.slice(j + 1, k) as Side
        const { seed } = JSON.parse(line.slice(k + 1))

        header.seed[side] = seed
        mark[side] = true
        break
      }
    }

    const { p1, p2, start } = mark
    if (start && p1 && p2) {
      return [header as Header, i + 1] as const
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
      const side = FOE[input.slice(j + 1) as Side]
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

        p = piped(line, 0)

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
