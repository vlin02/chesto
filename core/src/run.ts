import { Generation } from "@pkmn/data"
import { Patch, Preset } from "./version.js"
import { FoeUser } from "./client/user.js"
import { Observer } from "./client/observer.js"

export type Format = {
  gen: Generation
  patch: Patch
}

type MoveChoice =
  | {
      type: "struggle" | "recharge" | "stuck"
      moves: undefined
    }
  | {
      moves: string[]
      locked?: boolean
    }

export function getMoveChoice({ gen }: Format, obs: Observer): MoveChoice {
  const { active } = obs.ally

  let {
    volatiles: {
      "Encore": encore,
      "Taunt": taunt,
      "Heal Block": healBlock,
      "Locked Move": locked,
      "Disable": disable,
      "Throat Chop": throatChop,
      "Recharge": recharge,
      "Choice Locked": choiceLocked
    },
    item,
    lastMove
  } = active

  if (recharge)
    return {
      type: "recharge",
      moves: undefined
    }

  const { moveSet } = active
  const moves = []

  const lockedMove = locked?.move || choiceLocked?.move || encore?.move

  if (lockedMove) {
    if (!(lockedMove in moveSet))
      return {
        type: "stuck",
        moves: undefined
      }

    return { moves: [lockedMove], locked: !!locked }
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

    if (disable?.move === move) continue
    if (taunt && category === "Status") continue
    if (healBlock && heal) continue
    if (throatChop && sound) continue
    if (item === "Assault Vest" && category === "Status") continue

    moves.push(move)
  }

  if (!moves.length)
    return {
      type: "struggle",
      moves: undefined
    }

  return { moves }
}

export function getBaseForme({ gen, patch }: Format, forme: string) {
  return forme in patch ? forme : gen.species.get(forme)!.baseSpecies
}

export function getPotentialPresets(format: Format, user: FoeUser) {
  const { patch } = format

  const {
    base: { forme }
  } = user

  const baseForme = getBaseForme(format, forme)
  const presets = [...patch[baseForme].presets]

  if (baseForme === "Greninja") presets.push(...patch["Greninja-Bond"].presets)

  return presets
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
