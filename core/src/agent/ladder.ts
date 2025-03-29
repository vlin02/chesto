import { nanoid } from "nanoid"
import { Pool } from "undici"
import { loginAnon } from "../web/user.js"
import { WebSocket } from "ws"
import { Observer } from "../parser/observer.js"
import { Generations } from "@pkmn/data"
import { Dex } from "@pkmn/dex"
import { predict } from "./inference.js"
import { transportH } from "../model/state-h.js"
const inf = new Pool("http://172.31.57.228:8000")

const ws = new WebSocket("wss://sim3.psim.us/showdown/websocket")
const u = await loginAnon(ws, nanoid(10))

// const gen = new Generations(Dex).get(9)

// const rooms = new Map<
//   string,
//   {
//     obs: Observer
//     close: () => void
//   }
// >()

// u.global.on("search", (roomIds) => {
//   for (const roomId of roomIds) {
//     if (rooms.has(roomId)) continue

//     const obs = new Observer(gen)

//     let ready = false
//     const listener = async (lines: string[]) => {
//       let pending = false
//       for (const line of lines) {
//         pending ||= !!obs.read(line).pending
//       }

//       if (ready) {
//         const [{ action_id: choiceId }] = await predict(inf, "3/2-1742816842.pt", [
//           transportH.packBattle(transportH.encodeBattle(obs))
//         ])

//         u.send(roomId, "/" + obs.toInput(transportH.decodeChoice(obs, choiceId)))
//         ready = false
//       }

//       if (pending) ready = true
//     }

//     u.room.on(roomId, listener)

//     rooms.set(roomId, {
//       obs,
//       close: () => u.room.off(roomId, listener)
//     })
//   }
// })

// const _ = (async () => {
//   while (true) {
//     const req = await u.search("gen9randombattle")
//     await req()
//   }
// })()

// await new Promise<void>((r) => setTimeout(() => r(), 1000))
// ws.send("|/cancelsearch")
