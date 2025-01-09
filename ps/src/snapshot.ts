import { Battle } from "@pkmn/client"
import { Stats, StatusName } from "@pkmn/data"

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

const STATUSES = ["slp", "psn", "brn", "frz", "par", "tox"]

function statusToIndex(status: StatusName) {
  return STATUSES.indexOf(status)
}

function extractBattle(battle: Battle) {
  const {
    gens,
    gen,
    field,
    p1,
    p2,
    p3,
    p4,
    sides,
    turn,
    gameType,
    rated,
    rules,
    tier,
    teamPreviewCount,
    speciesClause,
    kickingInactive,
    totalTimeLeft,
    graceTimeLeft,
    lastMove,
    request,
    requestStatus
  } = battle

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
        return {
          id,
          ppLeft
        }
      })
    }
  }
}
