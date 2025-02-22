import {
  Choice,
  getAllOptions, Options,
  Run
} from "../run.js"

function optionsListed({ canTera, moves, switches }: Options): Choice[] {
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
  const opts = getAllOptions(run)
  const choices = optionsListed(opts)

  const i = Math.floor(Math.random() * choices.length)
  return choices[i]
}