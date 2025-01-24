import { Generation } from "@pkmn/data"
import { Observer } from "./observer/observer.js"
import { Preset } from "./version.js"
import { FoeUser } from "./observer/user.js"

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

  const { moveSet } = active
  const available = []

  for (const move in moveSet) {
    const {
      category,
      flags: { heal, sound }
    } = gen.moves.get(move)!

    const { used, max } = moveSet[move]
    if (used >= max) continue

    switch (move) {
      case "Stuff Cheeks": {
        if (!item?.endsWith("Berry")) continue
        break
      }
      case "Gigaton Hammer":
      case "Blood Moon": {
        if (lastMove === move) continue
      }
    }

    if (choiceLocked && choiceLocked.move in moveSet && choiceLocked.move !== move) continue
    if (lockedMove && lockedMove.move !== move) continue

    if (disable?.move === move && !lockedMove) continue
    if (encore && move !== encore.move) continue
    if (taunt && category === "Status") continue
    if (healBlock && heal) continue
    if (throatChop && sound) continue
    if (item === "Assault Vest" && category === "Status") continue

    available.push(move)
  }

  if (!available.length) return ["Struggle"]

  return available
}

export function matchesUser(preset: Preset, user: FoeUser) {
  const {
    base: { ability, moveSet },
    teraType
  } = user

  const { movepool, teraTypes, abilities } = preset
  if (teraTypes && teraType && !teraTypes.includes(teraType)) return false
  if (abilities && ability && !abilities.includes(ability)) return false
  // if (item && !items.includes(item)) return false
  if (!Object.keys(moveSet).every((move) => movepool.includes(move))) return false

  return true
}

export function getSetForme(gen: Generation, forme: string) {
  gen.species
}

// 1703173141 17b7ef1b1 {
//   role: 'Fast Support',
//   movepool: [ 'Transform' ],
//   teraTypes: [ 'Stellar' ]
// }

// 1703437168

// 1703722666

// 1703958831 58aa9c3a4 {
//   role: 'Fast Support',
//   movepool: [ 'Transform' ],
//   teraTypes: [ 'Stellar' ]
// }
// 1703960706 c144d28c8 {
//   role: 'Fast Support',
//   movepool: [ 'Transform' ],
//   teraTypes: [ 'Stellar' ]
// }
