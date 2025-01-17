import { Generation, Generations, Specie, Stats } from "@pkmn/data"
import { AllyMember, Observer } from "./parse/observer.js"
import { StatId, StatusId, TERRAIN_NAMES } from "./parse/protocol.js"
import { Dex } from "@pkmn/dex"

function scale(n: number, lo: number, hi: number) {
  return (n - lo) / (hi - lo)
}

export function extract(obs: Observer) {
  const gen = new Generations(Dex).get(9)
  const stats = new Stats(Dex)

  let weather = null
  let terrain = null
  const fields: { [k: string]: number } = {}

  if (obs.weather) {
    const { name, turn } = obs.weather
    weather = { name, turnsLeft: scale(5 - turn, 0, 5) }
  }

  for (const name in obs.fields) {
    const turn = obs.fields[name]
    const turnsLeft = scale(5 - turn, 0, 5)

    if (TERRAIN_NAMES.includes(name)) terrain = { name, turnsLeft }
    else fields[name] = turnsLeft
  }

  const { ally } = obs

  {
    let team = []
    const active = ally.active!

    for (const speciesName in ally.team) {
      const _member = ally.team[speciesName] as AllyMember
      const isActive = _member === active.member

      const {
        revealed,
        lvl,
        forme: formeName,
        gender,
        hp,
        ability,
        item,
        stats: _stats,
        status: _status,
        moveset: _moveset,
        teraType,
        oneTime: { "Battle Bond": battleBond, "Intrepid Sword": intrepidSword }
      } = _member

      const status: {
        id: StatusId | null
        sleepMovesLeft: [number, number]
        toxicTurns: number
      } = {
        id: null,
        sleepMovesLeft: [0, 0],
        toxicTurns: 0
      }

      if (_status) {
        const { id, move, turn } = _status
        status.id = id
        if (isActive) {
          if (id === "tox") status.toxicTurns = turn!
          if (id === "slp") status.sleepMovesLeft = [Math.min(1 - move!, 1), Math.min(3 - move!, 1)]
        }
      }

      const moveset = []
      for (const name in _moveset) {
        moveset.push({
          name,
          ppUsed: _moveset[name]
        })
      }

      let species = gen.species.get(speciesName)!
      let forme = gen.species.get(formeName)!

      if (species.cosmeticFormes?.some((x) => x === forme.name)) {
        forme = species
      }

      const stats: { [k in StatId]: number } = {}
      for (const id in _stats) stats[id] = scale(_stats[id], 0, 200)

      const transitionaryForme = toTransitionaryForme(forme)

      const member = {
        revealed,
        lvl,
        gender,
        hp,
        ability,
        item,
        stats,
        status,
        teraType,
        oneTime: {
          battleBond,
          intrepidSword
        },
        transitionaryForme
      }

      team.push(member)
    }
  }

  return {
    weather,
    terrain,
    fields
  }
}

function extractSpecies() {}

function extractMove(gen: Generation, id: string) {}
