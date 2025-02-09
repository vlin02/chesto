import { parseInput, split } from "./log.js"
import { Observer } from "./client/observer.js"
import { getMoveOption, getOptions, isTrapped, toMoves } from "./client/option.js"
import { FOE, Side } from "./client/protocol.js"
import { getPotentialPresets, matchesPreset } from "./version.js"
import { Format, Run } from "./run.js"
import { Replay } from "./db.js"

export function testSide(fmt: Format, replay: Replay, side: Side) {
  const { gen } = fmt

  const { inputs, outputs } = replay
  const obs = new Observer(gen)

  const run: Run = { fmt, obs }

  const opp = replay[FOE[side]]

  const hasZoroark = opp.team.some((x) => x.name === "Zoroark")

  for (let i = 0; i < outputs.length; i++) {
    const input = parseInput(inputs[inputs.length - outputs.length + i])
    const logs = outputs[i]

    if (input.type === "choose") {
      const { choice } = input
      if (input.side === side) {
        const { slots } = obs.ally

        const opt = getOptions(run)

        switch (choice.type) {
          case "move": {
            const { move } = choice

            if (opt.type !== "move") throw Error()

            const chosenMove =
              { recharge: "Recharge", struggle: "Struggle" }[move] ?? gen.moves.get(move)!.name
            const moves = toMoves(opt.move)

            if (!moves.includes(chosenMove)) throw Error()
            break
          }
          case "switch": {
            if (opt.type === "wait") throw Error()
            const { i } = choice
            const { species } = slots[i - 1]

            if (!opt.switches?.includes(species)) throw Error()
            break
          }
        }
      }
    }

    for (const msg of logs.flatMap((x) => split(x)[side])) {
      // console.log(msg)
      obs.read(msg)
    }

    if (!hasZoroark && obs.foe) {
      const { team } = obs.foe

      for (const species in team) {
        const user = team[species]

        const build = opp.team.find((x) => x.name === species)!
        const presets = getPotentialPresets(fmt, user)

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
        const [{ moveSlots, trapped }] = req.choices

        const aliveCnt = req.team.reduce((t, x) => t + (x.health ? 1 : 0), 0)

        const trappedA = aliveCnt !== 1 && isTrapped(active)
        const trappedB = !!trapped
        if (trappedA !== trappedB) throw Error()

        const movesA = toMoves(getMoveOption(run, active)).sort()
        const movesB = moveSlots
          .filter((x) => !x.disabled)
          .map((x) => x.name)
          .sort()
        if (JSON.stringify(movesA) !== JSON.stringify(movesB)) throw Error()

        for (const { name, pp, maxpp } of moveSlots) {
          const slot = active.moveSet[name]

          // outrage
          if (!pp) continue

          if (Math.max(0, slot.max - slot.used) !== pp) throw Error()
          if (slot.max !== maxpp) throw Error()
        }
      }

      for (let i = 0; i < 6; i++) {
        if (obs.ally.slots[i].species !== req.team[i].species) throw Error()
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

        if ((user === ally.active) !== active) throw Error()

        if (user.ability !== ability) throw Error()

        if (user.teraType !== teraType) throw Error()

        if (user.item !== item) throw Error()

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
