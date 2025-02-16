import { Run, isTrapped, getMoveOption, getValidSwitches, getValidRevives } from "../run.js"
import { MoveSlotFeature, extractMoveSlot } from "./obs.js"

export type Options = {
  canTera: boolean
  moves: MoveSlotFeature[]
  switches: string[]
}

export function extractOptions(run: Run): Options {
  const { obs } = run

  let canTera = false
  let moveSlots: MoveSlotFeature[] = []
  let switches: string[] = []

  const {
    req,
    ally: { active, isReviving, teraUsed }
  } = obs

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
      if (!trapped) switches = getValidSwitches(run)
      break
    case "switch":
      switches = isReviving ? getValidRevives(run) : getValidSwitches(run)
      break
  }

  return { canTera, moves: moveSlots, switches }
}
