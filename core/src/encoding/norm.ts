import { StatID } from "@pkmn/data"

export function scaleStat(id: StatID, n: number) {
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
