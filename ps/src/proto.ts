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
  | "???"
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

type Traits = {
  forme: string
  lvl: number
  gender: Gender
  tera: TypeName | null
}

export function parseTraits(s: string) {
  const parts = s.split(", ")
  const traits: Traits = { forme: parts[0], lvl: 100, gender: null, tera: null }

  for (let i = 1; i < parts.length; i++) {
    const part = parts[i]

    if (part === "M" || part === "F") {
      traits.gender = part
    } else if (part[0] === "L") {
      traits.lvl = Number(part.slice(1))
    } else if (part.startsWith("tera:")) {
      traits.tera = part.slice("tera:".length) as TypeName
    }
  }

  return traits
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

export function parseEffect(s: string) {
  let i = s.indexOf(": ")
  let item = undefined
  let ability = undefined
  let move = undefined

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

export function parseTypes(s: string) {
  return s.split("/") as TypeName[]
}

export function parseLabel(s: string) {
  const i = s.indexOf(": ")
  const side = s.slice(0, 2) as Side
  const species = s.slice(i + 2)
  return { side, species } as const
}
