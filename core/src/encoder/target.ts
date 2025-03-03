import { MoveTarget } from "@pkmn/data"

type SinglesTarget = "self" | "foe" | "field" | "allySide" | "foeSide" | "scripted"

const moveTargetMapping: Record<MoveTarget, SinglesTarget> = {
  // Moves that target the user in 1v1)

  // Moves that target the opponent
  adjacentFoe: "foe", // Single opponent is always adjacent in 1v1
  self: "self",
  adjacentAlly: "self", // No allies in 1v1, treated as self-only
  adjacentAllyOrSelf: "self", // No allies in 1v1, defaults to self
  allies: "self", // Only one active Pokemon, targets self
  allyTeam: "self", // Affects user's team (just self
  allAdjacentFoes: "foe", // Spread moves hit single opponent in 1v1
  randomNormal: "foe", // Random targeting among one foe is just foe
  normal: "foe", // Standard targeting always hits foe in 1v1
  any: "foe", // Free target choice becomes foe-only in 1v1
  allAdjacent: "foe", // Hits everything adjacent (both Pokemon in 1v1)

  // Field effects
  all: "field", // Affects entire field

  // Side conditions
  allySide: "allySide", // Adds a side condition (Light Screen, Reflect, etc.)
  foeSide: "foeSide", // Adds a side condition to opponent's side

  // Special cases
  scripted: "scripted" // Maintains special targeting logic
}

export function toSinglesTarget(t: MoveTarget) {
  return moveTargetMapping[t]
}
