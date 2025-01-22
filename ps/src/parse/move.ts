import { Generation } from "@pkmn/data"

export function getMaxPP(gen: Generation, move: string) {
  const { noPPBoosts, pp } = gen.moves.get(move)!
  return noPPBoosts ? pp : Math.floor(pp * 1.6)
}

export function isPressured(gen: Generation, move: string) {
  const {
    target,
    flags: { mustpressure }
  } = gen.moves.get(move)!

  return (
    mustpressure ||
    [
      "adjacentFoe",
      "all",
      "allAdjacent",
      "allAdjacentFoes",
      "any",
      "normal",
      "randomNormal",
      "scripted"
    ].includes(target)
  )
}

export function isLocking(gen: Generation, move: string) {
  const {
    self,
    flags: { charge }
  } = gen.moves.get(move)!

  return self?.volatileStatus === "lockedmove" || charge
}
