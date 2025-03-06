import { PRNGSeed } from "@pkmn/sim"
import { Side } from "./parser/protocol.js"
import { parseInput } from "./log.js"
import { StatId } from "./battle.js"
import { Role } from "./version.js"

export type Build = {
  name: string
  species: string
  gender: string | boolean
  moves: string[]
  ability: string
  evs: { [k in StatId]: number }
  ivs: { [k in StatId]: number }
  item: string
  level: number
  shiny: boolean
  nature?: string
  happiness?: number
  dynamaxLevel?: number
  gigantamax?: boolean
  teraType?: string
  role?: Role
}

export type Header = {
  formatId: string
  version: string
  versionOrigin?: string
  rated: boolean
  seed: {
    battle: PRNGSeed
    p1: PRNGSeed
    p2: PRNGSeed
  }
}

export function seekToStart(lines: string[], i: number) {
  let mark = { start: false, p1: false, p2: false, version: false }
  let header: any = { seed: {} }

  for (; i < lines.length; i++) {
    const line = lines[i]
    const input = parseInput(line)

    switch (input.type) {
      case "start": {
        const { formatId, seed, rated } = input
        header.seed.battle = seed
        header.formatId = formatId
        header.rated = rated
        mark.start = true
        break
      }
      case "version":
        header.version = input.hash
        break
      case "version-origin":
        header.versionOrigin = input.hash
        break
      case "player": {
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
