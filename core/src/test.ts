import { split } from "./log.js"
import { Observer } from "./client/observer.js"
import { getMoveOptions, isTrapped, toMoves } from "./client/action.js"
import { FOE, Side } from "./client/protocol.js"
import { Replay } from "./replay.js"
import { getPotentialPresets, matchesPreset } from "./version.js"
import { Format } from "./format.js"

export function testSide(format: Format, replay: Replay, side: Side) {
  const { gen } = format

  const { inputs, outputs } = replay
  const obs = new Observer(gen)

  const opp = replay[FOE[side]]

  const hasZoroark = opp.team.some((x) => x.name === "Zoroark")
  // if (JSON.stringify(outputs).includes("ability: Dancer")) {
  //   console.log("dancer")
  //   return
  // }

  for (let i = 0; i < outputs.length; i++) {
    const input = inputs[inputs.length - outputs.length + i]
    const logs = outputs[i]
    let j = 0

    let newReq = false

    // if (input.startsWith(`>${side}`)) {
    //   const [_, type, choice] = input.split(" ")
    //   const { active, slots, isReviving } = obs.ally

    //   switch (type) {
    //     case "move": {
    //       const moves = toMoves(getMoveOptions(format, active))
    //       if (
    //         !moves.includes(
    //           { recharge: "Recharge", struggle: "Struggle" }[choice] ?? gen.moves.get(choice)!.name
    //         )
    //       ) {
    //         console.log(moves, choice)
    //         throw Error()
    //       }
    //       break
    //     }
    //     case "switch": {
    //       const { species } = slots[Number(choice) - 1]
    //       const switches = getSwitchOptions(obs)

    //       if (isReviving ? obs.ally.team[species].hp[0] !== 0 : !switches.includes(species)) {
    //         throw Error()
    //       }

    //       if (obs.req.type !== "switch" && isTrapped(active)) {
    //         throw Error()
    //       }
    //       break
    //     }
    //   }
    // }

    for (const msg of logs.flatMap((x) => split(x)[side])) {
      // console.log(msg)
      const event = obs.read(msg)
      newReq ||= event === "request"
    }

    if (!hasZoroark && obs.foe) {
      const { team } = obs.foe

      for (const species in team) {
        const user = team[species]

        const build = opp.team.find((x) => x.name === species)!
        const presets = getPotentialPresets(format, user)

        if (
          !presets.some((preset) => {
            return preset.role === build.role && matchesPreset(preset, user)
          })
        ) {
          throw Error()
        }
      }
    }

    // if (obs.ally) {
    //   const { active } = obs.ally

    //   for (const n of ["Disable", "Encore", "Choice Locked", "Locked Move"] as const) {
    //     const x = active.volatiles[n]?.move
    //     if (x && !(x in active.moveSet)) throw [n, x, active.moveSet]
    //   }

    //   // const x = active.lastMove
    //   // if (x && !["Struggle", "Recharge"].includes(x) && !(x in active.moveSet))
    //   //   throw ["last move", x, active.moveSet]
    // }

    if (newReq) {
      j++
      const { ally, req: request } = obs

      if (request.type === "move") {
        const { active } = ally
        const [{ moveSlots, trapped }] = request.choices

        const moves = toMoves(getMoveOptions(format, active))

        if (request.team.filter((x) => !!x.health).length > 1 && !!trapped !== isTrapped(active)) {
          console.log(!!trapped, isTrapped(active), active)
          throw Error()
        }

        const expectedMoves = moveSlots
          .filter((x) => !x.disabled)
          .map((x) => x.name)
          .sort()

        if (
          !(moves.length === expectedMoves.length && moves.every((x) => expectedMoves.includes(x)))
        ) {
          console.log(ally.active)
          console.log(moves, expectedMoves)
          throw Error()
        }

        for (const { name, pp, maxpp } of moveSlots) {
          const slot = active.moveSet[name]

          if (
            (pp !== undefined && Math.max(0, slot.max - slot.used) !== pp) ||
            (maxpp !== undefined && slot.max !== maxpp)
          ) {
            console.log(name, pp, maxpp, slot)
            throw Error()
          }
        }
      }

      for (let i = 0; i < 6; i++) {
        if (obs.ally.slots[i].species !== request.team[i].species) {
          throw Error()
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
        stats,
        label: { gender, lvl }
      } of request.team) {
        const user = ally.team[species]

        for (const id of ["atk", "def", "spa", "spd", "spe"] as const) {
          if (user.stats[id] !== stats[id]) {
            throw Error()
          }
        }

        if ((user === ally.active) !== active) {
          throw Error()
        }

        if (user.ability !== ability) {
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

          if (!(user.hp[0] === hp[0] && user.hp[1] === hp[1])) {
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
