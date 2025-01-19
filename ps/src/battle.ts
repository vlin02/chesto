import { Generation, Move, Specie } from "@pkmn/data"
import { Observer } from "./parse/observer.js"

export function isPressureMove({ target, flags: { mustpressure } }: Move) {
  return (
    [
      "adjacentFoe",
      "all",
      "allAdjacent",
      "allAdjacentFoes",
      "any",
      "normal",
      "randomNormal",
      "scripted"
    ].includes(target) || mustpressure
  )
}

export function toTransitionaryForme({ baseSpecies, forme, name }: Specie) {
  switch (baseSpecies) {
    case "Minior": {
      return forme === "Meteor" ? name : baseSpecies
    }
    case "Terapagos":
      return forme === "Stellar" ? null : name
    case "Shaymin":
      return forme === "Sky" ? null : name
    case "Ogerpon":
      return ["Cornerstone-Tera", "Wellspring-Tera", "Hearthflame-Tera", "Teal-Tera"].includes(
        forme
      )
        ? null
        : name
    case "Eiscue":
    case "Cramorant":
    case "Mimikyu":
    case "Palafin":
    case "Meloetta":
    case "Morpeko": {
      return name
    }
    default:
      return null
  }
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

  const { moveSet } = active

  if (recharge) return ["Recharge"]

  const available = []

  if (lockedMove) {
    available.push(lockedMove.name)
  } else {
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

      if (disable?.move === name) {
        continue
      }
      if (choiceLocked && name !== choiceLocked.name) {
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
  }

  if (!available.length) return ["Struggle"]

  return available
}
