export type Log = ["update", string[]] | ["sideupdate", string] | ["end"]

export type Replay = {
  id: string
  uploadtime: number
  players: [string, string]
  rating: number
  private: number
  password: string | null
}

export type Rollout = {
  replay: Replay
  inputs: string[]
  logs: Log[]
}
