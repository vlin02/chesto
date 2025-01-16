import { TypeName } from "@pkmn/data"
import { join } from "path"
import { readFileSync } from "fs"

export type Set = {
  role: string
  hp: [number, number][]
  atk: number
  spe: number
  items: string[]
  abilities: string[]
  moves: string[]
  teraTypes: TypeName[]
}

export type Patch = {
  [k: string]: {
    level: number
    sets: Set[]
  }
}

const RELEASES = [
  { timestamp: 1715302353, hash: "64c595197014a5c73c7750f7a5da765daca4e830" },
  { timestamp: 1717083165, hash: "9f7f79d79b60902aabd3405dbe177101bcd28323" },
  { timestamp: 1717259560, hash: "e055d629b33bfc4f1e7e5c79f3aeb47f18ada00f" },
  { timestamp: 1718596438, hash: "05f7811d116fede574b72462be3d6ee6091e31ab" },
  { timestamp: 1719721671, hash: "5b24baa5a27cfc11fe69d971d5795ff1c043bf61" },
  { timestamp: 1719841709, hash: "77f6c9de2e9574bde1b78380997b03e94195a2af" },
  { timestamp: 1720764390, hash: "085839ca393b0a51fa8df78f0fe46bc5b4bedc89" },
  { timestamp: 1721089187, hash: "3049d4cf7ac6e66df0693a41d4c0ddce25b0a556" },
  { timestamp: 1722464891, hash: "df2036b01b64ad0f81ac518061f908351705a71b" },
  { timestamp: 1722464938, hash: "389b03e161b0b9aca5d26a3bd1aee5faacf78387" },
  { timestamp: 1723993193, hash: "f0bfd40ccf13e2de942973524a0569f63185ae7b" },
  { timestamp: 1725171953, hash: "7826b8b7d054368604e29175ec379f9c7ec7bd30" },
  { timestamp: 1725173666, hash: "1a96e9abe1ea080895700c6d9e3ae6957ac3a8b3" },
  { timestamp: 1726590847, hash: "604019454013eacf3d26daa472473ebe14f727f4" },
  { timestamp: 1727549486, hash: "2da507df8007fc3fc83484c84f7537fd09479966" },
  { timestamp: 1727706975, hash: "702a16a6574f5ab76ed692488a2aed8ddbd80499" },
  { timestamp: 1729705506, hash: "f2b538d8c2c87806bc0a047d25c20ed5c4084e5c" },
  { timestamp: 1730309386, hash: "f78ff14f21951244684e1e12c841f5061195f3a7" },
  { timestamp: 1731968003, hash: "5e8e7aee914c6161dfd262f313aa578a20defd36" },
  { timestamp: 1733014210, hash: "7df50ca2cc035b05b1fcb66fd91b5ab38524042c" },
  { timestamp: 1733069119, hash: "8510b48058cc2dde74cb6d5ccda277b8b8422da9" },
  { timestamp: 1733179787, hash: "8896008659a665011a18f478b3bc1cc73fe54f7f" },
  { timestamp: 1734365923, hash: "c6047ccfe6d036eebb74ca84639aeef836acc8f4" },
  { timestamp: 1735712534, hash: "5bd8b0d029759ffe623905eb4f7730c24f8dfccf" },
  { timestamp: 1735836690, hash: "ad28a307a33779dcc02d7234ec2b681bdcfebdf3" }
]

export function nearestSetHash(time: number) {
  for (let i = 1; i < RELEASES.length - 1; i++) {
    const { timestamp } = RELEASES[i]
    if (time < timestamp) return RELEASES[i - 1].hash
  }

  return RELEASES[RELEASES.length - 1].hash
}

export class PatchManager {
  dir: string
  setByHash: Map<string, Patch>
  constructor(dir: string) {
    this.dir = dir
    this.setByHash = new Map()
  }

  load(hash: string) {
    if (!this.setByHash.has(hash)) {
      this.setByHash.set(
        hash,
        JSON.parse(readFileSync(join(this.dir, `${hash}.json`), "utf-8")) as Patch
      )
    }
    return this.setByHash.get(hash)!
  }
}
