import { Choice, getValidActions, Actions, Run } from "../run.js"

function optionsListed({ tera: canTera, move: moves, switch: switches }: Actions): Choice[] {
  const choices: Choice[] = []
  switch (moves?.type) {
    case "struggle":
    case "recharge":
      choices.push({
        type: "move",
        move: { struggle: "Struggle", recharge: "Recharge" }[moves.type],
        tera: false
      })
      break
    case "default":
      const { moves: names } = moves
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

export function randomAgent(run: Run) {
  const opts = getValidActions(run)
  const choices = optionsListed(opts)

  const i = Math.floor(Math.random() * choices.length)
  return choices[i]
}

export function chioceToMessage(choice: Choice) {
  switch (choice.type) {
    case "move":
      const pfx = `move ${choice.move}`
      return choice.tera ? `${pfx} terastallize` : pfx
    case "switch":
      return `switch ${choice.species}`
  }
}
