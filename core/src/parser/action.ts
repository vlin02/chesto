import { Side } from "../battle.js"

export type Selection =
  | { type: "struggle" | "recharge"; tera?: undefined }
  | {
      type: "default"
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

export function toMoves(select: Selection) {
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
