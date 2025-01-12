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
export type WeatherName = "Snow" | "SunnyDay" | "SandStorm" | "RainDance"
export type Gender = "M" | "F" | null

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
  const [forme, lvl, gender] = s.split(", ") as [string, string | undefined, Gender | undefined]
  return { forme, lvl: lvl ? Number(lvl.slice(1)) : 100, gender: gender ?? null }
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

export function parseHp(s: string): [number, number] | null {
  if (s.slice(-3) === "fnt") return null
  const [a, b] = s.split(" ")[0].split("/")
  return [Number(a), Number(b)]
}

export function parseEntity(s: string) {
  let i = s.indexOf(": ")
  let item = null
  let ability = null
  let move = null

  switch (s.slice(0, i)) {
    case "item":
      item = s.slice(i + 2)
      break
    case "ability":
      ability = s.slice(i + 2)
      break
    case "move":
      move = s.slice(i + 2)
      break
  }

  return { item, ability, move, stripped: ability || move || item || s }
}
