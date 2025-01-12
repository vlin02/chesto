export type Decision =
  | {
      type: "switch"
      i: 0 | 1 | 2 | 3 | 4 | 5
    }
  | {
      type: "move"
      i: 0 | 1 | 2 | 3
      event?: "zmove" | "ultra" | "mega" | "dynamax" | "terastallize"
    }
  | {
      type: "auto"
    }

export function make(x: Decision) {
  switch (x.type) {
    case "switch": {
      const { i } = x
      return `switch ${i + 1}`
    }
    case "move": {
      const { i, event } = x
      let cmd = `move ${i + 1}`
      return event ? `${cmd} ${event}` : cmd
    }
    case "auto": {
      return "default"
    }
  }
}
