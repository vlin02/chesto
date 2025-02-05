import { FoeUser } from "./client/user.js"
import { Format } from "./client/actions.js"
import { Preset } from "./version.js"

export function getPresetForme({ gen, patch }: Format, forme: string) {
  return forme in patch ? forme : gen.species.get(forme)!.baseSpecies
}

export function getPotentialPresets(format: Format, user: FoeUser) {
  const { patch } = format

  const {
    base: { forme }
  } = user

  const baseForme = getPresetForme(format, forme)
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
