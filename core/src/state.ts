import { Generation, Generations, Specie, Stats } from "@pkmn/data"
import { Dex } from "@pkmn/dex"
import { AllyUser } from "./observer/user.js"
import { StatId, StatusId, TERRAIN_NAMES } from "./battle.js"
import { Observer } from "./observer/observer.js"

function scale(n: number, lo: number, hi: number) {
  return (n - lo) / (hi - lo)
}

export function toTransitionaryForme({ baseSpecies, forme, name }: Specie) {
  switch (baseSpecies) {
    case "Minior": {
      return forme === "Meteor" ? name : baseSpecies
    }
    case "Terapagos":
      return forme === "Stellar" ? null : name
    case "Shaymin":
      return forme === "" ? null : name
    case "Ogerpon":
      return ["Cornerstone-Tera", "Wellspring-Tera", "Hearthflame-Tera", "Teal-Tera"].includes(
        forme
      )
        ? null
        : name
    case "Eiscue":
      return forme === "Noice" ? null : name
    case "Palafin":
      return forme === "Hero" ? null : name
    case "Cramorant":
    case "Mimikyu":
    case "Meloetta":
    case "Morpeko": {
      return name
    }
    default:
      return null
  }
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
    for (const speciesName in ally.team) {
      const _user = ally.team[speciesName] as AllyUser
      const isActive = ally.active === _user

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
        moveSet: _moveset,
        teraType,
        flags: { battleBond, intrepidSword }
      } = _user

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
        const { id, turn, attempt } = _status
        status.id = id
        if (isActive) {
          if (id === "tox") status.toxicTurns = turn!
          if (id === "slp")
            status.sleepMovesLeft = [Math.min(1 - attempt!, 1), Math.min(3 - attempt!, 1)]
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

      const user = {
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

      team.push(user)
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
