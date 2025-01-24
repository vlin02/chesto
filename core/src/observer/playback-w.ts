import { Generations } from "@pkmn/data"
import { Dex } from "@pkmn/dex"
import { $ } from "execa"

//@ts-ignore
import { parentPort, workerData } from "worker_threads"

const { i, hash } = workerData as { i: number; hash: string }
const cwd = `/Users/vilin/ps/f${i}`

await $({ cwd })`git checkout ${hash}`
await $({ cwd })`npm run build`
await $({ cwd })`git reset --hard`

const Sim = await import(`ps${i}`)

const gen = new Generations(Dex).get(9)

const { randomSets: patch } = Sim.default.Teams.getGenerator("gen9randombattle")
for (const k in patch) {
  patch[k].presets = patch[k].sets
  delete patch[k].sets
  patch[gen.species.get(k)!.name] = patch[k]
  delete patch[k]
}

parentPort!.postMessage({
  hash,
  patch
})
