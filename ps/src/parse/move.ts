import { Move } from "@pkmn/data"

export type MoveSlot = {
  used: number
  max: number
}

export type MoveSet = { [k: string]: MoveSlot }

export function getMaxPP({ noPPBoosts, pp }: Move) {
  return noPPBoosts ? pp : Math.floor(pp * 1.6)
}