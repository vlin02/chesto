export type MoveSelection =
  | {
      type: "struggle" | "recharge"
    }
  | {
      type: "default"
      moves: string[]
      stuck?: boolean
    }

export type ActionSelection = {
  tera: boolean
  move: MoveSelection | null
  switch: string[]
}

export type Choice = 

export function flattenActions(select: ActionSelection) {
  const choices: Choice[] = []
  switch (select.move?.type) {
    case "struggle":
    case "recharge":
      choices.push({
        type: "move",
        move: { struggle: "Struggle", recharge: "Recharge" }[moves.type],
        tera: false
      })
      break
    case "default":
      const { moves } = select
      for (const name of names) {
        for (const tera of [false, true]) {
          if (tera && !canTera) continue
          choices.push({
            type: "move",
            move: name,
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
