import { Side } from "./protocol.js"

export type Log =
  | ["update", string[]]
  | [
      "sideupdate",
      (
        | {
            side: Side
            type: "error"
            message: string
          }
        | {
            side: Side
            type: "request"
            request: any
          }
      )
    ]
  | ["end", any]

export type Replay = {
  uploadtime: number
  id: string
  format: string
  players: [string, string]
  rating: number
  inputlog: string
  log: Log[]
  private: number
  password: string | null
}
