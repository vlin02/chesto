import { StatId } from "./battle.js"

export type Role =
  | ""
  | "Fast Attacker"
  | "Setup Sweeper"
  | "Wallbreaker"
  | "Tera Blast user"
  | "Bulky Attacker"
  | "Bulky Setup"
  | "Fast Bulky Setup"
  | "Bulky Support"
  | "Fast Support"
  | "AV Pivot"
  | "Doubles Fast Attacker"
  | "Doubles Setup Sweeper"
  | "Doubles Wallbreaker"
  | "Doubles Bulky Attacker"
  | "Doubles Bulky Setup"
  | "Offensive Protect"
  | "Bulky Protect"
  | "Doubles Support"
  | "Choice Item user"
  | "Z-Move user"
  | "Staller"
  | "Spinner"
  | "Generalist"
  | "Berry Sweeper"
  | "Thief user"

export type Member = {
  name: string
  species: string
  gender: string | boolean
  moves: string[]
  ability: string
  evs: { [k in StatId]: number }
  ivs: { [k in StatId]: number }
  item: string
  level: number
  shiny: boolean
  nature?: string
  happiness?: number
  dynamaxLevel?: number
  gigantamax?: boolean
  teraType?: string
  role?: Role
}

