import { Generation } from "@pkmn/data"
import { StatId, TypeName } from "../battle.js"

import { Health, Label, parseHealth, parseLabel, parseReference, Side } from "./protocol.js"

export type RawRequest = {
  side: {
    id: Side
    name: string
    pokemon: {
      ident: string
      details: string
      condition: string
      active: boolean
      stats: { [k in StatId]: number }
      baseAbility: string
      item: string
      ability: string
      moves: string[]
      teraType?: TypeName
      terastallized?: boolean
    }[]
  }
} & {
  active?: [
    {
      moves: [{ move: string; disabled: boolean; pp: number; maxpp: number }]
      trapped?: boolean
      canTerastallize?: boolean
    }
  ]
  forceSwitch?: boolean[]
  wait?: true
}

export type Member = {
  species: string
  label: Label
  health: Health | null
  stats: {
    [k in "atk" | "def" | "spa" | "spd" | "spe"]: number
  }
  active: boolean
  baseAbility: string
  ability: string | null
  item: string | null
  moves: string[]
  teraType?: TypeName
  terastallized?: boolean
}

export type Request = (
  | {
      type: "move"
      choices: [
        {
          moveSlots: { name: string; disabled: boolean; pp: number; maxpp: number }[]
          trapped?: boolean
          canTerastallize?: boolean
        }
      ]
    }
  | {
      type: "wait" | "switch"
    }
) & {
  side: Side
  name: string
  team: Member[]
}

export function parseRequest(
  gen: Generation,
  { active, forceSwitch: forceswitch, wait, side: { id, name, pokemon } }: RawRequest
): Request {
  const team: Member[] = []

  for (const {
    ident,
    details,
    condition,
    active,
    stats,
    baseAbility,
    item,
    ability,
    moves,
    teraType,
    terastallized
  } of pokemon) {
    const { species } = parseReference(ident)

    team.push({
      species,
      label: parseLabel(details),
      health: parseHealth(condition),
      stats,
      active,
      baseAbility,
      item: item === "" ? null : gen.items.get(item)!.name,
      ability: ability === "" ? null : gen.abilities.get(ability)!.name,
      moves: moves.map((x) => gen.moves.get(x)!.name),
      teraType,
      terastallized
    })
  }

  const base = { team, side: id, name }

  if (active) {
    const [{ moves, trapped, canTerastallize }] = active

    return {
      type: "move",
      choices: [
        {
          moveSlots: moves.map(({ move, maxpp, pp, disabled }) => {
            return {
              name: move,
              pp,
              maxpp,
              disabled
            }
          }),
          trapped,
          canTerastallize
        }
      ],
      ...base
    }
  }

  if (forceswitch) {
    return {
      type: "switch",
      ...base
    }
  }

  if (wait) {
    return {
      type: "wait",
      ...base
    }
  }

  throw Error()
}
