import { split } from "./log.js"
import { Observer } from "./client/observer.js"
import { getMoveOptions, getSwitchOptions, isTrapped, toMoves } from "./client/option.js"
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

  for (let i = 0; i < outputs.length; i++) {
    const input = inputs[inputs.length - outputs.length + i]
    const logs = outputs[i]

    if (input.startsWith(`>${side}`)) {
      const [_, type, choice] = input.split(" ")
      const { active, slots, isReviving } = obs.ally

      switch (type) {
        case "move": {
          const moves = toMoves(getMoveOptions(format, active))
          const chosenMove =
            { recharge: "Recharge", struggle: "Struggle" }[choice] ?? gen.moves.get(choice)!.name

          if (!moves.includes(chosenMove)) throw Error()
          break
        }
        case "switch": {
          const { species } = slots[Number(choice) - 1]
          const switches = getSwitchOptions(obs)

          if (isReviving) {
            if (obs.ally.team[species].hp[0] !== 0) throw Error()
          } else {
            if (!switches.includes(species)) throw Error()
          }

          if (obs.req.type !== "switch" && isTrapped(active)) throw Error()
          break
        }
      }
    }

    for (const msg of logs.flatMap((x) => split(x)[side])) {
      obs.read(msg)
    }

    if (!hasZoroark && obs.foe) {
      const { team } = obs.foe

      for (const species in team) {
        const user = team[species]

        const build = opp.team.find((x) => x.name === species)!
        const presets = getPotentialPresets(format, user)

        const buildFound = presets.some((preset) => {
          return preset.role === build.role && matchesPreset(preset, user)
        })

        if (!buildFound) throw Error()
      }
    }

    if (obs.req && obs.winner === undefined) {
      const { ally, req } = obs

      if (req.type === "move") {
        const { active } = ally
        const [{ moveSlots, trapped: _trapped }] = req.choices

        const moves = toMoves(getMoveOptions(format, active)).sort()

        const aliveCnt = req.team.reduce((t, x) => t + (x.health ? 1 : 0), 0)
        const trapped = aliveCnt !== 1 && isTrapped(active)
        if (trapped !== !!_trapped) throw Error()

        const expectedMoves = moveSlots
          .filter((x) => !x.disabled)
          .map((x) => x.name)
          .sort()

        if (JSON.stringify(moves) !== JSON.stringify(expectedMoves)) throw Error()

        for (const { name, pp, maxpp } of moveSlots) {
          const slot = active.moveSet[name]

          // outrage
          if (!pp) continue

          if (Math.max(0, slot.max - slot.used) !== pp) throw Error()
          if (slot.max !== maxpp) throw Error()
        }
      }

      for (let i = 0; i < 6; i++) {
        if (obs.ally.slots[i].species !== req.team[i].species) {
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
      } of req.team) {
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

          if (JSON.stringify(user.hp) !== JSON.stringify(hp)) throw Error()

          if (user.status?.id !== status) throw Error()
        } else {
          if (user.hp[0] !== 0) throw Error()
        }

        if (user.base.gender !== gender) throw Error()

        if (user.lvl !== lvl) throw Error()

        if (user.tera !== !!terastallized) throw Error()
      }
    }
  }
}
