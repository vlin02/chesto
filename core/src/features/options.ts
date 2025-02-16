import { Run, isTrapped, getMoveOption, getValidSwitches, getValidRevives } from "../run.js"
import { MoveSlotFeature, extractMoveSlot } from "./observer.js"

export type SwitchOptions = { [k: string]: boolean } 

export type Options = {
  canTera: boolean
  moves: MoveSlotFeature[]
  switches: SwitchOptions
}

export function extractOptions(run: Run): Options {
  const { obs } = run

  let canTera = false
  let moveSlots: MoveSlotFeature[] = []

  const {
    req,
    ally: { active, isReviving, teraUsed, team }
  } = obs

  let switches: { [k: string]: boolean } = {}

  for (const k in team) {
    switches[k] = false
  }

  switch (req.type) {
    case "move":
      const trapped = isTrapped(active)

      extractMoveSlot

      const moveOpt = getMoveOption(run, active)

      let moves: string[] = []

      if (moveOpt.type === "struggle") moves = ["Struggle"]
      if (moveOpt.type === "recharge") moves = ["Recharge"]
      if (moveOpt.type === "default") moves = moveOpt.moves

      moveSlots = moves.map((x) => extractMoveSlot(active.moveSet, x)!)
      if (!teraUsed && moveOpt.type === "default") canTera = true
      if (!trapped) {
        for (const k of getValidSwitches(run)){
          switches[k] = true
        }
      break
    case "switch":
      for(const k of isReviving ? getValidRevives(run) : getValidSwitches(run)) {
        switches[k] = true
      }

      break
  }

  return { canTera, moves: moveSlots, switches }
}
