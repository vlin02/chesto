import { scale } from "./norm.js"
import { ModStatId, MOVE_CATEGORY, StatusId, TYPE_NAMES } from "../battle.js"
import { Move, SecondaryEffect, StatusName } from "@pkmn/data"

const MOVE_FLAGS = [
  "bypasssub",
  "bite",
  "bullet",
  "charge",
  "contact",
  "dance",
  "defrost",
  "distance",
  "failcopycat",
  "failencore",
  "failinstruct",
  "failmefirst",
  "failmimic",
  "futuremove",
  "gravity",
  "heal",
  "metronome",
  "mirror",
  "mustpressure",
  "noassist",
  // "allyanim", not useful
  "nonsky",
  "noparentalbond",
  "nosketch",
  "nosleeptalk",
  "pledgecombo",
  "powder",
  "protect",
  "pulse",
  "punch",
  "recharge",
  "reflectable",
  "slicing",
  "snatch",
  "sound",
  "wind"
] as const

const SIDE_CONDITIONS = [
  "auroraveil",
  "lightscreen",
  "mist",
  "quickguard",
  "reflect",
  "safeguard",
  "spikes",
  "stealthrock",
  "stickyweb",
  "tailwind",
  "toxicspikes",
  "wideguard"
]

const VOLATILE_STATUSES = [
  "confusion",
  "flinch",
  "substitute",
  "saltcure",
  "glaiverush",
  "partiallytrapped",
  "leechseed",
  "sparklingaria",
  "mustrecharge",
  "healblock",
  "taunt",
  "encore",
  "disable",
  "curse",
  "lockedmove",
  "roost",
  "noretreat",
  "destinybond",
  "magnetrise",
  "protect",
  "yawn"
]

type UserEffect = {
  boosts: { [k in ModStatId]?: { n: number; p: number } }
  status?: { id: StatusId; p: number }
  volatileStatus?: { name: string; p: number }
}

type MoveEffect = {
  self: UserEffect
  opp: UserEffect
}

function withEffect(
  user: UserEffect,
  {
    chance: p = 1,
    boosts,
    status,
    volatileStatus
  }: {
    chance?: number
    boosts?: { [k in ModStatId]?: number }
    status?: StatusName
    volatileStatus?: string
  }
) {
  if (boosts) {
    for (const k in boosts) {
      user.boosts[k as ModStatId] = { n: boosts[k as ModStatId]!, p }
    }
  }

  if (status) user.status = { id: status, p }
  if (volatileStatus) user.volatileStatus = { name: volatileStatus, p }
}

export function reconcileEffect(move: Move) {
  const effect: MoveEffect = {
    self: { boosts: {} },
    opp: { boosts: {} }
  }

  const {
    target,
    selfBoost,

    self,
    secondaries
  } = move

  if (selfBoost?.boosts) withEffect(effect.self, selfBoost)
  if (self) withEffect(effect.self, self)

  withEffect(target === "self" ? effect.self : effect.opp, move)

  for (const secondary of secondaries ?? []) {
    withEffect(effect.opp, secondary)
    const { self } = secondary
    if (self) withEffect(effect.self, self)
  }

  return effect
}

export function encodeMove(move: Move) {
  const {
    // not useful
    exists,
    name,
    secondary,
    shortDesc,
    num,
    id,
    fullname,
    kind,
    effectType,
    gen,
    isNonstandard,
    isZ,
    isMax,
    maxMove,
    zMove,
    noPPBoosts,

    //only useful for curse
    nonGhostTarget,

    // derived
    stallingMove,

    // never true
    hasSheerForce,

    // later
    desc,

    // effect
    target,
    status,
    boosts,
    volatileStatus,

    self,
    selfBoost,

    secondaries,

    weather,
    terrain,
    pseudoWeather,
    slotCondition,

    selfSwitch,

    accuracy,
    basePower,
    category,
    pp,
    priority,
    flags,
    drain,
    type,
    critRatio,
    ignoreImmunity,
    ignoreAbility,
    condition: { duration, noCopy, affectsFainted } = {},
    recoil,
    sideCondition,
    overrideDefensiveStat,
    multihit,
    multiaccuracy,
    thawsTarget,
    forceSwitch,
    selfdestruct,
    heal,
    overrideOffensiveStat,
    overrideOffensivePokemon,
    hasCrashDamage,
    sleepUsable,
    callsMove,
    damage,
    ignoreEvasion,
    ignoreDefensive,
    breaksProtect,
    smartTarget,
    willCrit
  } = move

  const fMove: number[] = []

  fMove.push(scale(accuracy === true ? 100 : accuracy, 50, 100))
  fMove.push(scale(basePower, 0, 250))

  fMove.push(...MOVE_CATEGORY.map((x) => (category === x ? 1 : 0)))
  fMove.push(scale(pp, 0, 64))
  fMove.push(scale(priority, -7, 5))

  fMove.push(...MOVE_FLAGS.map((k) => (flags[k] ? 1 : 0)))
  fMove.push(drain ? drain[0] / drain[1] : 0)

  fMove.push(...TYPE_NAMES.map((x) => (type === x ? 1 : 0)))
  fMove.push([1 / 24, 1 / 8, 1 / 2, 1][willCrit ? 3 : critRatio!])

  fMove.push(heal ? heal[0] / heal[1] : 0)

  fMove.push(recoil ? recoil[0] / recoil[1] : 0)

  fMove.push(
    ...["Wish", "healingwish", "revivalblessing"].map((x) => (slotCondition === x ? 1 : 0))
  )

  fMove.push(...["always", "ifhit"].map((x) => (selfdestruct === x ? 1 : 0)))

  fMove.push(...SIDE_CONDITIONS.map((x) => (sideCondition === x ? 1 : 0)))
  fMove.push(...(Array.isArray(multihit) ? multihit : [multihit ?? 0, multihit ?? 0]))

  fMove.push(scale(duration ?? 0, 0, 5))

  fMove.push(...["level"].map((x) => (damage === x ? 1 : 0)))

  fMove.push(...["def"].map((x) => (overrideDefensiveStat === x ? 1 : 0)))
  fMove.push(
    ...["def", "target"].map((x) =>
      (overrideOffensiveStat ?? overrideOffensivePokemon) === x ? 1 : 0
    )
  )

  fMove.push(
    ...([true, "copyvolatile", "shedtail"] as const).map((x) => (selfSwitch === x ? 1 : 0))
  )

  fMove.push(
    ...["electricterrain", "grassyterrain", "mistyterrain", "psychicterrain"].map((x) =>
      terrain === x ? 1 : 0
    )
  )
  fMove.push(
    ...["fairylock", "gravity", "magicroom", "trickroom", "wonderroom"].map((x) =>
      pseudoWeather === x ? 1 : 0
    )
  )
  fMove.push(...["snow", "RainDance", "Sandstorm", "sunnyday"].map((x) => (weather === x ? 1 : 0)))
  fMove.push(
    ...[
      "auroraveil",
      "lightscreen",
      "mist",
      "quickguard",
      "reflect",
      "safeguard",
      "spikes",
      "stealthrock",
      "stickyweb",
      "tailwind",
      "toxicspikes",
      "wideguard"
    ].map((x) => (sideCondition === x ? 1 : 0))
  )

  fMove.push(
    ...[
      noCopy,
      affectsFainted,
      multiaccuracy,
      thawsTarget,
      forceSwitch,
      ignoreImmunity,
      ignoreAbility,
      ignoreEvasion,
      ignoreDefensive,
      breaksProtect,
      smartTarget,
      hasCrashDamage,
      sleepUsable,
      callsMove
    ].map((x) => (x ? 1 : 0))
  )

  return fMove
}
