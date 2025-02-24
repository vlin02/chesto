import { TypeName } from "@pkmn/data"
import { Gender, StatusId } from "../battle.js"

export type Side = "p1" | "p2"

export const SIDES = ["p1", "p2"] as const
export const FOE = { p1: "p2", p2: "p1" } as const

export type Label = {
  forme: string
  lvl: number
  gender: Gender
}

export function parseLabel(s: string) {
  const parts = s.split(", ")
  const label: Label = { forme: parts[0], lvl: 100, gender: null }

  for (let i = 1; i < parts.length; i++) {
    const part = parts[i]

    if (part === "M" || part === "F") {
      label.gender = part
    } else if (part[0] === "L") {
      label.lvl = Number(part.slice(1))
    }
  }

  return label
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

export type Health = { hp: [number, number]; status?: StatusId }

export function parseHealth(s: string): Health | null {
  if (s.slice(-3) === "fnt") return null
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

export function parseReference(s: string) {
  const i = s.indexOf(": ")
  const side = s.slice(0, 2) as Side
  const species = s.slice(i + 2)
  return { side, species } as const
}
