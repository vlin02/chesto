export function delimit(c: string) {
  return (s: string, i: number, n = 1) => {
    const args = []
    for (let _ = 0; _ !== n; _++) {
      let j = s.indexOf(c, i)

      args.push(s.slice(i, j === -1 ? s.length : j))

      i = j === -1 ? s.length : j + 1
      if (i === s.length) break
    }
    return { args, i }
  }
}

export const spaced = delimit(" ")
export const piped = delimit("|")
