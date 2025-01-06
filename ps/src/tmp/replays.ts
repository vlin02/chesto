// import { Observer } from "../client.js"
// import { seekToStart } from "../protocol.js"
// import { Replay, Log } from "../replay.js"
// import { VersionManager } from "../version.js"

// async function processBattle({ uploadtime }: Replay, inputs: string[]) {
//   let [{ formatId, ...seed }, i] = seekToStart(inputs, 0)

//   const obs = new Observer()

//   const registry = new VersionManager()
//   const Battle = await registry.setByUnixSeconds(uploadtime)
//   if (!Battle) throw Error()

//   let logs: Log[] = []

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
//     send: (...log) => logs.push(log as Log)
//   })

//   while (i < inputs.length) {
//     battle.sendUpdates()
//     obs.consume(logs)
//     logs = []

//     apply(battle, inputs[i])
//     i += 1
//   }
// }
