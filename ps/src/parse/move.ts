import { Generation, Move } from "@pkmn/data"
import { Observer } from "./observer.js"

export function isPressured({ target, flags: { mustpressure } }: Move) {
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

export function isLocked({ self, flags: {charge} }: Move) {
  return self?.volatileStatus === "lockedmove" || charge
}

export function availableMoves(gen: Generation, obs: Observer): string[] {
  const { active } = obs.ally

  let {
    volatiles: {
      "Encore": encore,
      "Taunt": taunt,
      "Heal Block": healBlock,
      "Locked Move": lockedMove,
      "Disable": disable,
      "Throat Chop": throatChop,
      "Recharge": recharge,
      "Choice Locked": choiceLocked
    },
    item,
    lastMove
  } = active

  if (recharge) return ["Recharge"]
  if (lockedMove) return [lockedMove.move]

  const { moveSet } = active

  const available = []

  for (const name in moveSet) {
    const {
      category,
      flags: { heal, sound }
    } = gen.moves.get(name)!

    const { used, max } = moveSet[name]
    if (used >= max) continue

    switch (name) {
      case "Stuff Cheeks": {
        if (!item?.endsWith("Berry")) continue
        break
      }
      case "Gigaton Hammer":
      case "Blood Moon": {
        if (lastMove === name) continue
      }
    }

    if (choiceLocked && name !== choiceLocked.move) continue

    if (disable?.move === name) {
      continue
    }
    if (encore && name !== encore.move) {
      continue
    }
    if (taunt && category === "Status") {
      continue
    }
    if (healBlock && heal) {
      continue
    }
    if (throatChop && sound) {
      continue
    }
    if (item === "Assault Vest" && category === "Status") {
      continue
    }

    available.push(name)
  }

  if (!available.length) return ["Struggle"]

  return available
}
