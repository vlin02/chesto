import { Patch } from "./version.js"
import { Log } from "./log.js"
import { Observation } from "./__tmp/encoder/observation.js"
import { Collection, Db } from "mongodb"
import { Side } from "./battle.js"
import { Choice } from "./parser/option.js"
import { Build } from "./replay.js"

export type Player = {
  name: string
  team: Build[]
}

export type Step = {
  side: Side
  observation: Observation
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

type DB = {
  versions: Collection<Version>
  replays: Collection<Replay>
}

export function withSchema(db: Db): DB {
  return {
    versions: db.collection("versions"),
    replays: db.collection("replays")
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
