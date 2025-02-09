import { ObjectId } from "mongodb"
import { Log } from "./log.js"
import { Build } from "./build.js"
import { ID, toID } from "@pkmn/data"
import { PRNGSeed } from "@pkmn/sim"
import { Side, FOE } from "./client/protocol.js"

export type Player = {
  name: string
  team: Build[]
}

export type Replay = {
  id: string
  version: string
  uploadtime: number
  rating: number
  private: number
  password: string | null
  inputs: string[]
  outputs: Log[][]
  p1: Player
  p2: Player
}

export type Header = {
  formatId: ID
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
      case ">version-origin":
        header.versionOrigin = line.slice(j + 1)
        break
      case ">version":
        header.version = line.slice(j + 1)
        mark.version = true
        break
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

export interface BattleLike {
  choose(type: string, input: string): void
  win(side: Side | null): void
  sendUpdates(): void
}

export function apply(battle: BattleLike, input: string) {
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
