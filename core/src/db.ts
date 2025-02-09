import { Collection, Db } from "mongodb"
import { Patch } from "./version.js"
import { Log } from "./log.js"
import { Build } from "./build.js"

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

export type Version = {
  hash: string
  timestamp: number
  patch: Patch
}

type Move = {
  name: string
  f: number[]
  desc: {
    [k: string]: number[]
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

export type DB = {
  replays: Collection<Replay>
  versions: Collection<Version>
  moves: Collection<Move>
}

export function toDB(db: Db): DB {
  return {
    replays: db.collection("replays"),
    versions: db.collection("versions"),
    moves: db.collection("moves")
  }
}
