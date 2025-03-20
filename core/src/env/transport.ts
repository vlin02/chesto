import { BattleF } from "./model/state.js"

function toBuf(x: any, type: "float" | "int"): Buffer {
  x = x.flat(Infinity)
  if (type === "float") x = new Float32Array(x)
  else x = new Int32Array(x)
  return Buffer.from(x.buffer)
}

export type PackedBattle = {
  partyEnc: Buffer
  userEnc: Buffer
  activeIdx: Buffer
  moveChoiceIdx: Buffer
  moveMask: Buffer
  switchMask: Buffer
}

export function packBinary({
  partyEnc,
  userEnc,
  activeIdx,
  moveChoiceIdx,
  moveMask,
  switchMask
}: BattleF) {
  return {
    partyEnc: toBuf(partyEnc, "float"),
    userEnc: toBuf(userEnc, "float"),
    activeIdx: toBuf(activeIdx, "int"),
    moveChoiceIdx: toBuf(moveChoiceIdx, "int"),
    moveMask: toBuf(moveMask, "int"),
    switchMask: toBuf(switchMask, "int")
  }
}
