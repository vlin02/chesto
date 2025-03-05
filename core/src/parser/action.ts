export type Selection =
  | {
      type: "struggle" | "recharge"
    }
  | {
      type: "default"
      moves: string[]
      stuck?: boolean
    }

export type Action = {
  tera: boolean
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

  switch (select?.type) {
    case "struggle":
    case "recharge":
      choices.push({
        type: "move",
        move: { struggle: "Struggle", recharge: "Recharge" }[select.type],
        tera: false
      })
      break
    case "default":
      const { moves } = select
      for (const move of moves) {
        for (const tera of [false, true]) {
          if (tera && !action.tera) continue

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
