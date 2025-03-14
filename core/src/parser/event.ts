import { Side } from "../battle.js"

type Event = {
  winner?: Side | "tie"
  turn?: number
  req?: Request
  error?: string
}
