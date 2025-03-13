import { Side } from "../battle.js"

export type Selection = {
  tera: boolean
  moves: string[]
  stuck?: boolean
}

export type Option = {
  select: Selection | null
  switches: string[]
}

export type Choice =
  | {
      type: "select"
      move: string
      tera: boolean
    }
  | {
      type: "switch"
      species: string
    }

export type Action = { side: Side; choice: Choice }

export function toChoices(option: Option): Choice[] {
  const { select, switches } = option
  const choices: Choice[] = []

  if (select) {
    const { moves } = select
    for (const move of moves) {
      for (const tera of [true, false]) {
        if (!select.tera && tera) continue
        choices.push({
          type: "select",
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

export function formatChoice(choice: Choice) {
  switch (choice.type) {
    case "select":
      const pfx = `move ${choice.move}`
      return choice.tera ? `${pfx} terastallize` : pfx
    case "switch":
      return `switch ${choice.species}`
  }
}
