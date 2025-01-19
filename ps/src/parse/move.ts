import { Volatiles } from "./volatile.js"

export type MoveSlot = {
  used: number
  max: number
}

export class MoveSet {
  volatiles: Volatiles
  slots: { [k: string]: MoveSlot }

  constructor(volatiles: Volatiles) {
    this.volatiles = volatiles
    this.slots = {}
  }

  add(move: string, max: number) {
    this.slots[move] = {
      used: 0,
      max
    }
  }

  get(move: string) {
    return (this.volatiles["Transform"]?.moves ?? this.slots)[move]
  }
}
