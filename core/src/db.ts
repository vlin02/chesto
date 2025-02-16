import { Collection, Db } from "mongodb"
import { Patch } from "./version.js"
import { Log } from "./log.js"
import { Build } from "./build.js"
import { BattleFeature } from "./features/observer.js"
import { Choice } from "./run.js"
import { Options } from "./features/options.js"

export type Player = {
  name: string
  team: Build[]
}

export type Sample = {
  battle: BattleFeature
  options: Options
  choice: Choice
}

export type Step = {
  input: string
  logs: Log[]
  sample: Sample | null
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
  samples: (Sample | null)[]
  steps: Step[]
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
  f: number[]
  desc: {
    mistral: number[]
  }
}

type Ability = {
  name: string
  desc: {
    mistral: number[]
  }
}

type Item = {
  name: string
  desc: {
    mistral: number[]
  }
}

export async function createReplays(db: Db, name: string) {
  await db.createCollection(name, {
    storageEngine: { wiredTiger: { configString: "block_compressor=zstd" } }
  })

  const col = db.collection(name)
  await col.createIndex({ rating: 1 })
  await col.createIndex({ uploadtime: 1 })
  await col.createIndex({ id: 1 })
  await col.createIndex({ format: 1 })

  return col
}

export async function createVersions(db: Db, name: string) {
  await db.createCollection(name, {
    storageEngine: { wiredTiger: { configString: "block_compressor=zstd" } }
  })

  const col = db.collection(name)
  return col
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

type Type = {
  name: string
  x: number[]
  num: number
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
