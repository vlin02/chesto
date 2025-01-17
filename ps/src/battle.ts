import { Generation, Specie } from "@pkmn/data"
import { Observer } from "./obs.js"

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
  const {
    volatiles: {
      "Transform": transform,
      "Encore": encore,
      "Taunt": taunt,
      "Heal Block": healBlock,
      "Locked Move": lockedMove,
      "Disable": disable,
      "Throat Chop": throatChop,
      "Recharge": recharge
    },
    member,
    lastMove
  } = obs.active("ally")

  let { moveset, item } = member

  if (transform?.complete) moveset = transform.moveset
  if (recharge) return ["Recharge"]
  
  const available = []
  for (const name in moveset) {
    const {
      pp,
      category,
      basePower,
      flags: { heal, sound }
    } = gen.moves.get(name)!
    if (moveset[name] >= pp) continue

    if (disable?.move === name) {
      // console.log("=== name")
      continue
    }
    if (lockedMove && name !== lockedMove.move) {
      // console.log(".move")
      continue
    }
    if (encore && name !== encore.move) {
      // console.log(".move")
      continue
    }
    if (taunt && category === "Status") {
      // console.log("Status")
      continue
    }
    if (healBlock && heal) {
      // console.log("&& heal")
      continue
    }
    if (throatChop && sound) {
      // console.log("&& sound")
      continue
    }
    if (item === "Assault Vest" && category === "Status") {
      // console.log("=== 0")
      continue
    }

    if (
      item &&
      ["Choice Band", "Choice Specs", "Choice Scarf"].includes(item) &&
      lastMove &&
      lastMove !== name
    )
      continue

    available.push(name)
  }

  return available
}
