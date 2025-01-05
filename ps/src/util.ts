type Comparable = { valueOf: () => number | string } | number | string

export function compare<T, V extends Comparable[]>(fn: (x: T) => V, dir: (1 | -1)[] = []) {
  return (a: T, b: T) => {
    const [x, y] = [a, b].map(fn)

    for (let i = 0; i < x.length; i++) {
      const d = dir[i] ?? 1
      if (x[i] < y[i]) return -1 * d
      if (y[i] < x[i]) return 1 * d
    }

    return 0
  }
}

export function partition<T>(arr: T[], size: number) {
  const bins: T[][] = []
  for (let i = 0; i < arr.length; i += size) {
    bins.push(arr.slice(i, i + size))
  }
  return bins
}
