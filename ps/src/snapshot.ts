import { Battle, WeatherName } from "@pkmn/client"
import { BoostID, BoostsTable, HitEffect, Stats, StatusName } from "@pkmn/data"

function scale(lo: number, hi: number, v: number, neg = false) {
  if (neg) lo = (lo + hi) / 2
  return (v - lo) / (hi - lo)
}

const TYPES = [
  "Normal",
  "Fighting",
  "Flying",
  "Poison",
  "Ground",
  "Rock",
  "Bug",
  "Ghost",
  "Steel",
  "Fire",
  "Water",
  "Grass",
  "Electric",
  "Psychic",
  "Ice",
  "Dragon",
  "Dark",
  "Fairy",
  "Stellar"
] as const

const VOLATILES = [
  "leechseed",
  "futuresight",
  "protosynthesis",
  "quarkdrive",
  "substitute",
  "taunt",
  "yawn",
  "confusion",
  "charge",
  "throatchop",
  "encore",
  "disable",
  "flashfire",
  "magnetrise",
  "slowstart",
  "doomdesire",
  "attract",
  "saltcure",
  "noretreat"
] as const
type Volatile = keyof typeof VOLATILES

type Type = keyof typeof TYPES

type Effect = {
  boosts: { [k in BoostID]?: [number, number] }
  status?: [StatusName, number]
  volatile?: [string, number]
  sideCondition?: string
}

function applyBoosts(effect: Effect, boosts: Partial<BoostsTable>, chance = 1) {
  for (const [k, v] of Object.entries(boosts)) {
    effect.boosts[k as BoostID] = [v, chance]
  }
}

function applyHitEffect(
  effect: Effect,
  { boosts, volatileStatus: volatile }: HitEffect,
  chance = 1
) {
  if (boosts) {
    applyBoosts(effect, boosts, chance)
  }

  if (volatile) {
    effect.volatile = [volatile, 1]
  }
}

function extractBattle(battle: Battle) {
  const { gens, gen, field, p1, p2 } = battle

  const stats = new Stats(gens.dex)

  for (const {
    // CONSIDERED
    level,

    teraType,
    canTerastallize,
    terastallized,

    hp,
    status,
    fainted,
    boosts,
    gender,

    statusState,
    volatiles,

    item: itemId,
    trapped,
    maybeTrapped,

    moveSlots,
    types,
    addedType,
    switching,

    isGrounded,
    effectiveAbility,
    isActive,

    speciesForme,

    // NOT BEING CONSIDERED

    // being computed via stats
    maxhp,

    // always 0
    statusStage,

    // only differs on dynamax
    baseMaxhp,

    // never true
    maybeDisabled,

    // only applies to zoroark
    illusion,
    revealedDetails,

    // used to calculate item
    lastItem,

    // not very relevant
    itemEffect,
    lastItemEffect,
    lastMove,
    lastMoveTargetLoc,
    movesUsedWhileActive,
    timesAttacked,
    moveThisTurn,
    hurtThisTurn,
    weighthg,

    // not applicable to randbat
    teamPreviewItem,

    // contextual
    side,
    slot,

    // cosmetic
    shiny,
    name,
    hpcolor,
    // what is this for each team member?
    originalIdent,
    // make sure all of these details are included in the poke ?
    searchid,
    // does opp team update details ?
    details,

    // species forme - forme changes
    baseSpeciesForme,

    // prefer speciesForme
    species,
    baseSpecies,

    // set not available
    set,
    ivs,
    evs,
    happiness,
    hpType,
    nature,

    // same as slot
    position,
    // not useful in 1v1
    ident,

    // covered by effective ability
    ability,
    baseAbility,

    // used by types
    actualTypes,

    // encoded by switching
    newlySwitched,
    beingCalledBack,

    // not applicable to gen 9
    maxMoves,
    zMoves,
    canDynamax,
    canGigantamax,
    canMegaEvo,
    canUltraBurst,

    isTerastallized,

    // derived from moveslots
    moves,

    //derived from this.item
    hasItem
  } of p1.team) {
    const { baseStats } = gen.species.get(speciesForme)!
    const { sleepTurns, toxicTurns } = statusState

    const abilityId = effectiveAbility()

    return {
      level: scale(level, 76, 100),
      hp: scale(hp, 200, 400),
      gender,
      stats: Object.fromEntries(
        (["hp", "atk", "def", "spa", "spd", "spe"] as const).map((k) => {
          return [k, scale(stats.calc(k, baseStats.hp, 31, 85, level), 100, 300)]
        })
      ),
      teraType: teraType! as string as Type,
      terastallized: !!terastallized,
      canTerastallize: !!canTerastallize,
      status: status ?? null,
      fainted,
      boosts: Object.fromEntries(
        Object.entries(boosts).map(([k, v]) => {
          return [k, scale(v, -6, 6, true)]
        })
      ),
      sleepTurnsLeft: status === "slp" ? [1 - sleepTurns, 3 - sleepTurns] : [0, 0],
      toxicTurns,
      itemId: itemId ?? null,
      abilityId: abilityId ?? null,
      maybeTrapped: !!maybeTrapped,
      trapped: !!trapped,
      types: types as string[] as Type[],
      addedType: addedType ? (addedType as string as Type) : null,
      switching: switching ?? null,
      grounded: isGrounded(),
      active: isActive(),
      volatiles: Object.keys(volatiles).filter((v) => VOLATILES.includes(v as any)) as Volatile[],

      moves: moveSlots.map(({ id, ppUsed }) => {
        const {
          basePower,
          accuracy,
          pp,
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
          noPPBoosts,

          // NOT CONSIDERED

          // covered by secondaries
          secondary,

          // only used by curse, but not very useful unless given the description
          nonGhostTarget,

          // only applies in doubles & specifically dragon darts
          smartTarget
        } = gen.moves.get(id)!

        const effects: { self: Effect; foe: Effect } = {
          self: {
            boosts: {}
          },
          foe: {
            boosts: {}
          }
        }

        if (selfBoost) {
          const { boosts } = selfBoost
          if (boosts) {
            applyBoosts(effects.self, boosts)
          }
        }

        if (self) {
          applyHitEffect(effects.self, self)
        }

        if (secondaries) {
          for (const secondary of secondaries) {
            const { chance, self } = secondary
            if (!chance) continue
            if (self) {
              applyHitEffect(effects.self, self, chance)
            } else {
              applyHitEffect(effects.foe, secondary, chance)
            }
          }
        }

        if (boosts) {
          switch (target) {
            case "self":
              applyBoosts(effects.self, boosts, 1)
              break
            default:
              throw Error(target)
          }
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
              effects.self.sideCondition = sideCondition
              break
            case "foeSide":
              effects.foe.sideCondition = sideCondition
              break
            default:
              throw Error(target)
          }
        }

        return {
          id,
          ppLeft: pp - ppUsed,
          accuracy: scale(accuracy === true ? 100 : accuracy, 0, 100),
          category,

          terrain,

          // attack
          critRatio: critRatio ?? 1,
          basePower: scale(basePower, 0, 200),
          damageByLevel: damage === "level",
          statOverride: {
            defensive: overrideDefensiveStat || overrideDefensivePokemon || null,
            offensive: overrideOffensiveStat || overrideOffensivePokemon || null
          },
          multiHit: {
            times: Array.isArray(multihit) ? multihit : [multihit ?? 0, multihit ?? 0],
            recheckAcc: !!multiaccuracy
          },
          willCrit: !!willCrit,

          effects: effects,

          // heal % of user hp
          heal: heal ? heal[0] / heal[1] : 0,

          // steal % of foe hp
          drain: drain ? drain[0] / drain[1] : 0,

          thawsTarget: !!thawsTarget,
          recoil: recoil ? recoil[0] / recoil[1] : 0,
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
          },
          type,
          priority
        }
      })
    }
  }
}
