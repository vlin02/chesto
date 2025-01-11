import { Battle, Pokemon, TerrainName, WeatherName } from "@pkmn/client"
import { BoostID, BoostsTable, Generation, HitEffect, Move, Stats, StatusName } from "@pkmn/data"
import { Side, SIDES } from "./battle.js"
import { View } from "./view.js"

type Effect = {
  boosts: { [k in BoostID]?: [number, number] }
  status?: [StatusName, number]
  volatile?: [string, number]
  sideCondition?: string
}

function applyBoosts(effect: Effect, boosts: Partial<BoostsTable>, p = 1) {
  for (const [k, v] of Object.entries(boosts)) {
    effect.boosts[k as BoostID] = [v, p]
  }
}

function applyHitEffect(
  effect: Effect,
  { boosts, status, volatileStatus: volatile }: HitEffect,
  p = 1
) {
  if (boosts) {
    applyBoosts(effect, boosts, p)
  }

  if (status) {
    effect.status = [status, p]
  }

  if (volatile) {
    effect.volatile = [volatile, p]
  }
}

export function extractBattle(b: Battle, v: View, [ally, foe]: Side[]) {
  const [p1, p2] = [b[ally], b[foe]]

  const {
    field: { weather, weatherState, terrain, terrainState, pseudoWeather }
  } = b

  const { gen } = b
  const stats = new Stats(gen.dex)

  return {
    weather: weather
      ? {
          id: weatherState.id as WeatherName,
          turnsLeft: [weatherState.minDuration, weatherState.maxDuration]
        }
      : null,
    terrain: terrain
      ? {
          id: terrainState.id as TerrainName,
          turnsLeft: [terrainState.minDuration, terrainState.maxDuration]
        }
      : null,
    pseudoWeathers: Object.values(pseudoWeather).map((x) => {
      return {
        id: x.id,
        turnsLeft: [x.minDuration, x.maxDuration]
      }
    }),
    ally: {
      team: p1.team.slice(-6).map((pkmn) => {
        return {
          details: extractPokemon(gen, stats, pkmn, v),
          moves: pkmn.moveSlots.map((m) => {
            const m1 = gen.moves.get(m.id)!
            return {
              id: m.id,
              details: extractMove(gen.moves.get(m.id)!),
              ppLeft: m1.pp - m.ppUsed
            }
          })
        }
      }),
      sideConditions: Object.values(p1.sideConditions).map(({ name, minDuration, maxDuration }) => {
        return {
          id: name,
          turnsLeft: [minDuration, maxDuration]
        }
      })
    },
    foe: {
      team: p2.team.slice(-6).map((pkmn) => {
        return {
          details: extractPokemon(gen, stats, pkmn, v),
          moves: pkmn.moveSlots.map((m) => {
            const m1 = gen.moves.get(m.id)!
            return {
              id: m.id,
              details: extractMove(gen.moves.get(m.id)!),
              ppLeft: m1.pp - m.ppUsed
            }
          })
        }
      }),
      sideConditions: Object.values(p2.sideConditions).map(({ name, minDuration, maxDuration }) => {
        return {
          id: name,
          turnsLeft: [minDuration, maxDuration]
        }
      })
    }
  }
}

export function extractPokemon(gen: Generation, stats: Stats, pkmn: Pokemon, view: View) {
  const {
    level,

    teraType,

    hp,
    maxhp,
    status,
    fainted,
    boosts,
    gender,

    statusState,
    volatiles,

    item: itemId,
    trapped,
    maybeTrapped,

    types,
    addedType,
    switching,

    speciesForme
  } = pkmn

  const side = view[SIDES[pkmn.side.n]]

  const { baseStats } = gen.species.get(speciesForme)!
  const { sleepTurns, toxicTurns } = statusState

  const abilityId = pkmn.effectiveAbility()

  return {
    id: pkmn.baseSpecies.id,
    active: side.active === pkmn.name,
    status: status ?? null,
    fainted,
    boosts,
    hpLeft: maxhp === 0 ? 1 : hp / maxhp,
    teraType: teraType ?? null,
    terastallized: side.terastallized?.[0] === pkmn.name,

    sleepTurnsLeft: status === "slp" ? [1 - sleepTurns, 3 - sleepTurns] : [0, 0],
    toxicTurns,
    maybeTrapped: !!maybeTrapped,
    trapped: !!trapped,
    switching: switching ?? null,
    revealed: side.revealed.has(pkmn.name),

    grounded: pkmn.isGrounded(),

    level,
    gender,
    stats: Object.fromEntries(
      (["hp", "atk", "def", "spa", "spd", "spe"] as const).map((k) => {
        return [k, stats.calc(k, baseStats[k], 31, 85, level)]
      })
    ),
    itemId: itemId ?? null,
    abilityId: abilityId ?? null,
    types,
    addedType: addedType ?? null,
    volatiles: Object.keys(volatiles)
  }
}

export function extractMove(move: Move) {
  const {
    basePower,
    accuracy,
    category,
    type,
    priority,
    target,
    flags,

    sideCondition,
    terrain,
    pseudoWeather,
    weather: weatherId,

    drain,
    critRatio,
    ignoreImmunity,

    secondaries,
    recoil,
    self,
    selfSwitch,
    overrideDefensiveStat,
    multiaccuracy,
    multihit,
    thawsTarget,
    stallingMove,
    forceSwitch,
    selfdestruct,
    heal,
    overrideOffensiveStat,
    overrideOffensivePokemon,
    overrideDefensivePokemon,
    hasCrashDamage,
    sleepUsable,
    callsMove,
    selfBoost,
    damage,
    ignoreDefensive,
    ignoreEvasion,
    breaksProtect,
    ignoreAbility,
    willCrit,
    noPPBoosts
  } = move

  const effects: { ally: Effect; foe: Effect } = {
    ally: {
      boosts: {}
    },
    foe: {
      boosts: {}
    }
  }

  if (selfBoost) {
    const { boosts } = selfBoost
    if (boosts) {
      applyBoosts(effects.ally, boosts)
    }
  }

  if (self) {
    applyHitEffect(effects.ally, self)
  }

  if (secondaries) {
    for (const secondary of secondaries) {
      const { chance, self } = secondary
      if (!chance) continue
      if (self) {
        applyHitEffect(effects.ally, self, chance / 100)
      } else {
        applyHitEffect(effects.foe, secondary, chance / 100)
      }
    }
  }

  switch (target) {
    case "self":
      applyHitEffect(effects.ally, move, 1)
      break
    case "normal":
      applyHitEffect(effects.foe, move, 1)
      break
    default:
      break
  }

  let weather: WeatherName | null = null

  if (weatherId) {
    weather = (
      {
        RainDance: "Rain",
        sunnyday: "Sun",
        snow: "Snow"
      } as const
    )[weatherId as string]!
  }

  if (sideCondition) {
    switch (target) {
      case "allySide":
        effects.ally.sideCondition = sideCondition
        break
      case "foeSide":
        effects.foe.sideCondition = sideCondition
        break
      default:
        throw Error(target)
    }
  }

  return {
    accuracy: accuracy === true ? 1 : accuracy / 100,
    category,
    type,
    priority,

    flags: Object.keys(flags),
    weather,
    pseudoWeather: pseudoWeather ?? null,
    terrain: terrain ?? null,

    critRatio: category === "Status" ? 0 : critRatio,
    basePower,
    damageByLevel: damage === "level",
    statOverride: {
      defensive: overrideDefensiveStat || overrideDefensivePokemon || null,
      offensive: overrideOffensiveStat || overrideOffensivePokemon || null
    },
    multiHit: multihit
      ? {
          times: Array.isArray(multihit) ? multihit : [multihit, multihit],
          recheckAcc: !!multiaccuracy
        }
      : null,
    willCrit: !!willCrit,

    effects,

    heal: heal ? heal[0] / heal[1] : 0,

    drain: drain ? drain[0] / drain[1] : 0,

    recoil: recoil ? recoil[0] / recoil[1] : 0,
    thawsTarget: !!thawsTarget,
    hasCrashDamage: !!hasCrashDamage,
    sleepUsable: !!sleepUsable,
    selfSwitch: !!selfSwitch,
    forceSwitch: !!forceSwitch,
    noPPBoosts: !!noPPBoosts,
    selfDestruct: !!selfdestruct,
    stallingMove: !!stallingMove,
    callsMove: !!callsMove,
    breaksProtect: !!breaksProtect,
    ignore: {
      immunity: !!ignoreImmunity,
      defensive: !!ignoreDefensive,
      evasion: !!ignoreEvasion,
      ability: !!ignoreAbility
    }
  }
}
