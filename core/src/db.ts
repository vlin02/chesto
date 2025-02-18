import { Collection, Db } from "mongodb"
import { Patch } from "./version.js"
import { Log } from "./log.js"
import { Build } from "./build.js"
import { Observation } from "./features/observation.js"
import { Choice } from "./run.js"
import { Options } from "./features/options.js"
import { Side } from "./client/protocol.js"

export type Player = {
  name: string
  team: Build[]
}

export type Step = {
  side: Side
  observation: Observation
  options: Options
  choice: Choice
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
  steps: (Step | null)[]
  p1: Player
  p2: Player
}

export type Version = {
  hash: string
  timestamp: number
  patch: Patch
}

type Move = {
  name: string
  i: number
  x: number[]
  desc: {
    mistral: number[]
  }
}

type Ability = {
  name: string
  i: number
  desc: {
    mistral: number[]
  }
}

type Item = {
  name: string
  i: number
  desc: {
    mistral: number[]
  }
}

type Type = {
  name: string
  x: number[]
  i: number
}

export type DB = {
  replays: Collection<Replay>
  versions: Collection<Version>
  moves: Collection<Move>
  items: Collection<Item>
  abilities: Collection<Ability>
  types: Collection<Type>
}

export function withSchema(db: Db): DB {
  return {
    replays: db.collection("replays"),
    versions: db.collection("versions"),
    moves: db.collection("moves"),
    items: db.collection("items"),
    abilities: db.collection("abilities"),
    types: db.collection("types")
  }
}

export class VersionCache {
  cache: Map<string, Version>

  constructor(public db: DB) {
    this.cache = new Map()
  }

  async load(hash: string) {
    let ver = this.cache.get(hash)
    if (!ver) {
      ver = (await this.db.versions.findOne({ hash }))!
      this.cache.set(hash, ver)
    }
    return ver
  }
}
