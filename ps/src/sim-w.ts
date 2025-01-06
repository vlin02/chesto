// import { parentPort } from "worker_threads"
// import { VersionManager } from "./version.js"
// import { seekToStart, apply } from "./protocol.js"
// import { Replay } from "./replay.js"

// parentPort!.on("message", async (line: string) => {
//   const replay: Replay = JSON.parse(line)
//   const { uploadtime, inputlog } = replay

//   const inputs = inputlog.split("\n")
//   const log: any[] = []

//   let [{ formatId, ...seed }, i] = seekToStart(inputs, 0)

//   const registry = new VersionManager()
//   const Battle = await registry.setByUnixSeconds(uploadtime)
//   if (!Battle) throw Error()

//   const battle = new Battle({
//     formatid: formatId,
//     seed: seed.battle,
//     p1: {
//       name: "p1",
//       seed: seed.p1
//     },
//     p2: {
//       name: "p2",
//       seed: seed.p2
//     },
//     send: (...x) => log.push(x)
//   })

//   while (i < inputs.length) {
//     apply(battle, inputs[i])
//     i += 1
//     battle.sendUpdates()
//   }

//   parentPort!.postMessage(JSON.stringify({ ...replay, log }))
// })
