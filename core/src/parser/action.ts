export type MoveSelection =
  | {
      type: "struggle" | "recharge"
    }
  | {
      type: "default"
      moves: string[]
      stuck?: boolean
    }

export type ActionSelection = {
  tera: boolean
  move: MoveSelection | null
  switch: string[]
}
