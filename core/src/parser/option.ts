
export type MoveSelection =
  | { type: "struggle" | "recharge"; tera?: undefined }
  | {
      type: "default"
      tera: boolean
      moves: string[]
      stuck?: boolean
    }

export type Option = {
  select: MoveSelection | null
  switches: string[]
}

export type Choice =
  | {
      type: "move"
      move: string
      tera: boolean
    }
  | {
      type: "switch"
      species: string
    }


export function toMoves(select: MoveSelection) {
  if (select.type === "default") return select.moves
  return [{ struggle: "Struggle", recharge: "Recharge" }[select.type]]
}

export function toChoices(option: Option): Choice[] {
  const { select, switches } = option
  const choices: Choice[] = []

  if (select) {
    const moves = toMoves(select)
    for (const move of moves) {
      for (const tera of [true, false]) {
        if (tera && !(select.type === "default" && select.tera)) continue
        choices.push({
          type: "move",
          move,
          tera
        })
      }
    }
  }

  for (const species of switches) {
    choices.push({
      type: "switch",
      species
    })
  }

  return choices
}
