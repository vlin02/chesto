import { Generation, Generations } from "@pkmn/data"
import { split } from "../log.js"
import { Observer } from "./observer.js"
import { availableMoves, getPotentialPresets, matchesPreset, Run } from "../run.js"
import { Patch, Version } from "../version.js"
import { FOE, Side } from "./protocol.js"
import { Replay } from "../replay.js"
import { Dex } from "@pkmn/dex"
import { MongoClient } from "mongodb"
import { Request } from "./request.js"

type Env = {
  gen: Generation
  patches: Map<string, Patch>
}

function testSide({ gen, patches }: Env, replay: Replay, side: Side) {
  const { version, outputs } = replay
  const obs = new Observer(gen)

  const run: Run = {
    obs,
    gen,
    patch: patches.get(version)!
  }

  const opp = replay[FOE[side]]

  const canTestPresets = !opp.team.some((x) => x.name === "Zoroark")

  for (const logs of outputs) {
    let req: Request | null = null

    for (const msg of logs.flatMap((x) => split(x)[side])) {
      // console.log(msg)
      const event = obs.read(msg)
      if (event === "request") req = obs.request
    }

    if (canTestPresets && obs.foe) {
      const { team } = obs.foe

      for (const species in team) {
        const user = team[species]

        const build = opp.team.find((x) => x.name === species)!
        const presets = getPotentialPresets(run, user)

        if (
          !presets.some((preset) => {
            return preset.role === build.role && matchesPreset(preset, user)
          })
        ) {
          throw Error()
        }
      }
    }

    if (req) {
      const { ally } = obs

      if (req.type === "move") {
        const [{ moveSlots }] = req.choices

        const moves = availableMoves(run, true)
        const expectedMoves = moveSlots
          .filter((x) => !x.disabled)
          .map((x) => x.name)
          .sort()

        if (
          !(moves.length === expectedMoves.length && moves.every((x) => expectedMoves.includes(x)))
        ) {
          throw Error()
        }

        for (const { name, pp, maxpp } of moveSlots) {
          const slot = ally.active.moveSet[name]

          if (
            (pp !== undefined && Math.max(0, slot.max - slot.used) !== pp) ||
            (maxpp !== undefined && slot.max !== maxpp)
          ) {
            throw Error()
          }
        }
      }

      for (const {
        item,
        active,
        ability,
        teraType,
        terastallized,
        species,
        health,
        label: { gender, lvl }
      } of req.team) {
        const user = ally.team[species]

        if ((user === ally.active) !== active) {
          throw Error()
        }

        if (user.ability !== ability) {
          console.log(user)
          console.log(ability)
          throw Error()
        }

        if (user.teraType !== teraType) {
          throw Error()
        }

        if (user.item !== item) {
          throw Error()
        }

        if (health) {
          const { hp, status } = health

          if (!(user.hp[0] == hp[0] && user.hp[1] === hp[1])) {
            throw Error()
          }

          if (!(user.status?.id === status)) {
            throw Error()
          }
        } else {
          if (!(user.hp[0] === 0)) {
            throw Error()
          }
        }

        if (!(user.base.gender === gender)) {
          throw Error()
        }

        if (user.lvl !== lvl) {
          throw Error()
        }

        if (user.tera !== !!terastallized) {
          throw Error()
        }
      }
    }
  }
}

const gen = new Generations(Dex).get(9)

const mongo = new MongoClient("mongodb://localhost:27017")
await mongo.connect()

const db = mongo.db("chesto")
const Replays = db.collection<Replay>("replays")
const Versions = db.collection<Version>("versions")

const env: Env = {
  gen,
  patches: new Map((await Versions.find().toArray()).map((x) => [x.hash, x.patch]))
}

let j = 0
for await (const replay of Replays.find(
  {
    // id: "gen9randombattle-2003220640",
    // id: "gen9randombattle-2013995086",
    // id: "gen9randombattle-2016459202"
    // id: "gen9randombattle-2019636837"
    // id: "gen9randombattle-2039204208",
    // id: "gen9randombattle-2041558770"
    // id: "gen9randombattle-2061737964"
    // id: "gen9randombattle-2077043659"
  },
  {
    // skip: 27000
  }
)) {
  const { id } = replay
  try {
    testSide(env, replay, "p2")
  } catch (e) {
    console.log(id)
    throw e
  }

  if (++j % 1000 === 0) console.log(j)
}

await mongo.close()
