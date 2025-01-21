import { ChoiceRequest } from "./protocol.js"

export function checkLocked({ active }: ChoiceRequest, move: string) {
  if (!active) return undefined
  const [{ moves }] = active

  return moves.every((x) => x.disabled || (!x.disabled && x.move === move))
}
