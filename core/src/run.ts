import { Generation } from "@pkmn/data"
import { Observer } from "./client/observer.js"
import { Patch, Preset } from "./version.js"
import { FoeUser } from "./client/user.js"

export type Run = {
  gen: Generation
  patch: Patch
  obs: Observer
}

export function availableMoves({ gen, obs }: Run, matchProtocol = false): string[] {
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

  if (!matchProtocol && choiceLocked && !(choiceLocked.move in moveSet)) {
    return []
  }

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

export function matchesPreset(preset: Preset, user: FoeUser) {
  const {
    base: { ability, item, moveSet },
    teraType
  } = user

  const {
    movepool,
    agg: { moves, teraTypes, abilities, items }
  } = preset

  if (teraTypes && teraType && !teraTypes.includes(teraType)) return false
  if (abilities && ability && !abilities.includes(ability)) return false
  if (item && !items.includes(item)) return false
  if (!Object.keys(moveSet).every((move) => moves.includes(move) || movepool.includes(move)))
    return false

  return true
}

export function getPotentialPresets({ gen, patch }: Run, user: FoeUser) {
  const {
    base: { forme }
  } = user

  let baseForme = forme in patch ? forme : gen.species.get(forme)!.baseSpecies
  const presets = [...patch[baseForme].presets]

  if (baseForme === "Greninja") presets.push(...patch["Greninja-Bond"].presets)

  return presets
}
