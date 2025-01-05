import { Side } from "./protocol.js"
import { Request } from "@pkmn/protocol"

export type Log = ["update", string[]] | ["sideupdate", string] | ["end"]

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
