export const STAT_RANGES = {
  hp: [191, 566],
  atk: [13, 348],
  def: [57, 393],
  spa: [85, 318],
  spd: [71, 402],
  spe: [57, 357]
} as const

export function scale(n: number, lo: number, hi: number, neg = false) {
  if (neg) {
    const mid = (hi + lo) / 2
    return scale(n, mid, hi)
  }
  return (n - lo) / (hi - lo)
}
