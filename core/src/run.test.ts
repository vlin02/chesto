import { split } from "./log.js"
import { Observer } from "./client/observer.js"
import { availableMoves, Format, getPotentialPresets, matchesPreset } from "./run.js"
import { FOE, Side } from "./client/protocol.js"
import { Replay } from "./replay.js"
import { STAT_IDS } from "./battle.js"

export function testSide(format: Format, replay: Replay, side: Side) {
  const { gen } = format

  const { outputs } = replay
  const obs = new Observer(gen)

  const opp = replay[FOE[side]]

  const hasZoroark = opp.team.some((x) => x.name === "Zoroark")

  for (const logs of outputs) {
    let newRequest = false

    for (const msg of logs.flatMap((x) => split(x)[side])) {
      // console.log(msg)
      const event = obs.read(msg)
      newRequest ||= event === "request"
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

    if (newRequest) {
      const { ally, request } = obs

      if (request.type === "move") {
        const [{ moveSlots }] = request.choices

        const moves = availableMoves(format, obs, true)
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
        stats,
        label: { gender, lvl }
      } of request.team) {
        const user = ally.team[species]

        for (const id of STAT_IDS) {
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
