import { Generation, Specie } from "@pkmn/data"
import { Observer } from "./obs.js"
import { CHOICE_ITEMS } from "./proto.js"

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
      "Recharge": recharge,
    },
    member,
    lastMove,
    choiceItemMove
  } = obs.active("ally")

  let { moveset, item } = member

  if (transform?.complete) moveset = transform.moveset
  if (recharge) return ["Recharge"]

  const available = []
  for (const name in moveset) {
    const {
      pp,
      category,
      flags: { heal, sound }
    } = gen.moves.get(name)!
    if (moveset[name] >= Math.floor(pp * 1.6)) continue

    switch (name) {
      case "Stuff Cheeks": {
        if (!item?.endsWith("Berry")) continue
      }
    }

    if (disable?.move === name) {
      //      console.log("=== name")
      continue
    }
    if (item && CHOICE_ITEMS.includes(item) && choiceItemMove && choiceItemMove !== name) {
      //console.log("choice")
      continue
    }
    if (lockedMove && name !== lockedMove.move) {
      //console.log(".move")
      continue
    }
    if (lockedMove && name !== lockedMove.move) {
      //console.log(".move")
      continue
    }
    if (encore && name !== encore.move) {
      //console.log(".move")
      continue
    }
    if (taunt && category === "Status") {
      //console.log("Status")
      continue
    }
    if (healBlock && heal) {
      //console.log("&& heal")
      continue
    }
    if (throatChop && sound) {
      //console.log("&& sound")
      continue
    }
    if (item === "Assault Vest" && category === "Status") {
      //console.log("=== 0")
      continue
    }

    available.push(name)
  }

  if (!available.length) return ["Struggle"]

  return available
}
