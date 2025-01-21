import { TypeName } from "@pkmn/data"
import { Gender, StatId, StatusId } from "./species.js"

export type Side = "p1" | "p2"

export const SIDES = ["p1", "p2"] as const
export const FOE = { p1: "p2", p2: "p1" } as const

export type Member = {
  ident: string
  details: string
  condition: string
  active: boolean
  stats: { [k in StatId]: number }
  baseAbility: string
  item: string
  ability: string
  moves: string[]
  teraType?: TypeName
  terastallized?: boolean
}

export type ChoiceRequest = {
  side: {
    id: Side
    name: string
    pokemon: Member[]
  }
} & {
  active?: [
    {
      moves: [{ move: string; disabled: boolean; pp: number; maxpp: number }]
      trapped?: boolean
      canTerastallize?: boolean
    }
  ]
  forceswitch?: boolean[]
  wait?: true
}

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

export type Traits = {
  forme: string
  lvl: number
  gender: Gender
}

export function parseTraits(s: string) {
  const parts = s.split(", ")
  const traits: Traits = { forme: parts[0], lvl: 100, gender: null }

  for (let i = 1; i < parts.length; i++) {
    const part = parts[i]

    if (part === "M" || part === "F") {
      traits.gender = part
    } else if (part[0] === "L") {
      traits.lvl = Number(part.slice(1))
    }
  }

  return traits
}

export function parseTags(strs: string[]) {
  const tags: { [k: string]: string } = {}

  for (const s of strs) {
    const i = s.indexOf("]")
    const name = s.slice(1, i)
    const value = s.slice(i + 1).trim()
    tags[name] = value
  }

  return tags
}

export function parseCondition(
  s: string
): { hp: null; status: undefined } | { hp: [number, number]; status?: StatusId } {
  if (s.slice(-3) === "fnt") return { hp: null, status: undefined }
  let [frac, status] = s.split(" ")

  return {
    hp: frac.split("/").map(Number) as [number, number],
    status: status as undefined | StatusId
  }
}

export function parseEntity(s: string | undefined) {
  let item = undefined
  let ability = undefined
  let move = undefined
  let pokemon = undefined

  if (s) {
    let i = s.indexOf(": ")
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
      case "pokemon":
        pokemon = s.slice(i + 2)
        break
    }
  }

  return { item, ability, move, stripped: ability || move || item || s || "" }
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
