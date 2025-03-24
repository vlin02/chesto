import { Battle, PRNGSeed, toID } from "@pkmn/sim"
import { Log } from "./log.js"

export type BattleSeed = {
  battle: PRNGSeed
  p1: PRNGSeed
  p2: PRNGSeed
}

export function startBattle({
  formatId,
  seed,
  p1,
  p2,
  send
}: {
  formatId: string
  seed?: BattleSeed
  p1?: string
  p2?: string
  send?: (log: Log) => void
}) {
  return new Battle({
    formatid: toID(formatId),
    seed: seed?.battle,
    p1: { name: p1, seed: seed?.p1 },
    p2: { name: p2, seed: seed?.p2 },
    send: (...log) => send?.(log as Log)
  })
}