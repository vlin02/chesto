import { FOE, piped, Side, SIDES } from "./client/protocol.js"

export type Log = ["update", string[]] | ["sideupdate", string] | ["end", string]

export function split(log: Log) {
  let p

  const chs: { p1: string[]; p2: string[] } = { p1: [], p2: [] }

  const [type] = log

  switch (type) {
    case "update": {
      const [, lines] = log

      let i = 0
      while (i < lines.length) {
        const line = lines[i]

        p = piped(line, 0)

        if (p.args[0] === "split") {
          p = piped(line, p.i)
          const side = p.args[0] as Side

          chs[side].push(lines[i + 1])
          chs[FOE[side]].push(lines[i + 2])

          i += 3
        } else {
          for (const side of SIDES) chs[side].push(line)
          i += 1
        }
      }
      break
    }
    case "sideupdate": {
      const [, line] = log
      const side = line.slice(0, 2) as Side
      const msg = line.slice(3)

      chs[side].push(msg)
      break
    }
    case "end": {
      const [, line] = log
      const msg = `|end|${line}`
      for (const side of SIDES) chs[side].push(msg)
      break
    }
  }

  return chs
}
