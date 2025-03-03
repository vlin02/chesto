import { Patch } from "./version.js"
import { Log } from "./log.js"
import { Build } from "./build.js"
import { Observation } from "./encoder/observation.js"
import { Choice } from "./run.js"
import { Options } from "./encoder/options.js"
import { Side } from "./client/protocol.js"
import { Db } from "mongodb"

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

export class VersionCache {
  cache: Map<string, Version>

  constructor(public db: Db) {
    this.cache = new Map()
  }

  async load(hash: string) {
    let ver = this.cache.get(hash)
    if (!ver) {
      ver = (await this.db.collection<Version>("versions").findOne({ hash }))!
      this.cache.set(hash, ver)
    }
    return ver
  }
}
