import { Generation, Generations, Stats } from "@pkmn/data"
import { Dex } from "@pkmn/dex"
import { Observer } from "./client/observer.js"

function scale(n: number, lo: number, hi: number) {
  return (n - lo) / (hi - lo)
}

export function asInterimForme(gen: Generation, id: string) {
  const { baseSpecies, name, forme } = gen.species.get(id)!

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
    case "Palafin":
      return forme === "Hero" ? null : name
    case "Eiscue":
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

  const { ally: _ally, foe: _foe, turn, weather, fields } = obs

  let ally
  {
    const { teraUsed, fields, active, wish, team: _team } = _ally
    let team: { [k: string]: any } = {}

    for (const species in _ally.team) {
      const _user = _ally.team[species]

      const {
        revealed,
        lvl,
        hp,
        formeChange,
        item,
        base,
        stats,
        status,
        teraType,
        flags,
        lastMove,
        lastBerry,
        volatiles,
        boosts,
        tera
      } = _user

      team[species] = {
        revealed,
        lvl,
        hp,
        formeChange,
        item,
        base,
        stats,
        status,
        teraType,
        flags,
        lastMove,
        lastBerry,
        volatiles,
        boosts,
        tera,
        interimForme: asInterimForme(gen, _user.forme)
      }
    }

    ally = { teraUsed, active: active.species, fields, wish, team }
  }

  const stats = new Stats(Dex)

  let foe
  {
  }

  return {
    ally,
    weather,
    fields,
    turn
  }
}

function extractSpecies() {}

function extractMove(gen: Generation, id: string) {}
