export type Side = "p1" | "p2"
export const SIDES = ["p1", "p2"] as const

export const FOE = { p1: "p2", p2: "p1" } as const

export type StatId = "atk" | "def" | "spa" | "spd" | "spe"
export type BoostId = "atk" | "def" | "spa" | "spd" | "spe" | "evasion" | "accuracy"

export type TypeName =
  | "Normal"
  | "Fighting"
  | "Flying"
  | "Poison"
  | "Ground"
  | "Rock"
  | "Bug"
  | "Ghost"
  | "Steel"
  | "Fire"
  | "Water"
  | "Grass"
  | "Electric"
  | "Psychic"
  | "Ice"
  | "Dragon"
  | "Dark"
  | "Fairy"
  | "Stellar"

export type StatusId = "slp" | "psn" | "brn" | "frz" | "par" | "tox"
export type WeatherId = "Snow" | "SunnyDay" | "SandStorm" | "RainDance"

export function piped(s: string, i: number, n = 1) {
  const args = []
  for (let j = 0; j !== n; j++) {
    let k = s.indexOf("|", i + 1)
    if (k === -1) k = s.length
    args.push(s.slice(i + 1, k))
    i = k
    if (i === s.length) break
  }
  return { args, i }
}

export function parseTraits(s: string) {
  const [species, lvl, gender] = s.split(", ") as [
    string,
    string | undefined,
    "M" | "F" | undefined
  ]
  return { species, lvl: lvl ? Number(lvl.slice(1)) : 100, gender: gender ?? null }
}

export function parseTags(strs: string[]) {
  const tags: { [k: string]: string } = {}

  for (const s of strs) {
    const i = s.indexOf("]")
    const name = s.slice(1, i)
    const value = s.slice(i + 2)
    tags[name] = value
  }

  return tags
}
