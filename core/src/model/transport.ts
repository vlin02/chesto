import { Move } from "@pkmn/data";
import { Observer } from "../parser/observer.js";
import { Choice } from "../parser/option.js";

export interface Transport<T> {
  getMoveFeat: (move: Move) => any 
  getBattleFeat: (obs: Observer) => T
  packBattle: (x: T) => any
  decodeChoice: (obs: Observer, id: number) => Choice
}

export function toArrayBuffer(x: any, type: "float" | "int"): Buffer {
  x = x.flat(Infinity)
  if (type === "float") x = new Float32Array(x)
  else x = new Int32Array(x)
  return Buffer.from(x.buffer)
}
