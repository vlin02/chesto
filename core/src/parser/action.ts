export type Selection = {
  tera: boolean
  moves: string[]
  stuck?: boolean
}

export type Action = {
  select: Selection | null
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

export function toChoices(action: Action): Choice[] {
  const { select, switches } = action
  const choices: Choice[] = []

  if (select) {
    const { moves } = select
    for (const move of moves) {
      for (const tera of [true, false]) {
        if (!select.tera && tera) continue
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
