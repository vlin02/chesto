import { StatID } from "@pkmn/data"

export const STAT_MAX = {
  hp: 714,
  atk: 348,
  def: 393,
  spa: 318,
  spd: 402,
  spe: 35
} as const

export function scaleStat(_: StatID, n: number) {
  // max stat is 714 (hp)
  return n / 714
}

export function scalePP(n: number) {
  return n / 64
}

export function scalePower(n: number) {
  return n / 250
}

export function scale(n: number, lo: number, hi: number, neg = false) {
  if (neg) {
    const mid = (hi + lo) / 2
    return scale(n, mid, hi)
  }
  return (n - lo) / (hi - lo)
}
