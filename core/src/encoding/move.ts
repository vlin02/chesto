import { scale } from "./norm.js"
import { MOVE_CATEGORY, TYPE_NAMES } from "../battle.js"
import { Move } from "@pkmn/data"
import { encodeMoveEffect, reconcileEffect } from "./effect.js"

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
  "allyanim",
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

const PSEUDO_WEATHERS = ["fairylock", "gravity", "magicroom", "trickroom", "wonderroom"]
const WEATHERS = ["snow", "RainDance", "Sandstorm", "sunnyday"]
const TERRAINS = ["electricterrain", "grassyterrain", "mistyterrain", "psychicterrain"]

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
    overrideDefensivePokemon,
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

  const f: number[] = []

  f.push((accuracy === true ? 100 : accuracy) / 100)
  f.push(basePower / 250)

  f.push(...MOVE_CATEGORY.map((x) => (category === x ? 1 : 0)))
  f.push(scale(pp, 0, 64))
  f.push(scale(priority, -7, 5))

  f.push(...MOVE_FLAGS.map((k) => (flags[k] ? 1 : 0)))
  f.push(drain ? drain[0] / drain[1] : 0)

  f.push(...TYPE_NAMES.map((x) => (type === x ? 1 : 0)))
  f.push([1 / 24, 1 / 8, 1 / 2, 1][willCrit ? 3 : critRatio!])

  f.push(heal ? heal[0] / heal[1] : 0)

  f.push(recoil ? recoil[0] / recoil[1] : 0)

  f.push(...["always", "ifhit"].map((x) => (selfdestruct === x ? 1 : 0)))

  f.push(...SIDE_CONDITIONS.map((x) => (sideCondition === x ? 1 : 0)))
  f.push(...(Array.isArray(multihit) ? multihit : [multihit ?? 0, multihit ?? 0]))

  f.push(scale(duration ?? 0, 0, 5))

  f.push(...["level"].map((x) => (damage === x ? 1 : 0)))

  f.push(
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

  f.push(...TERRAINS.map((x) => (terrain === x ? 1 : 0)))
  f.push(...PSEUDO_WEATHERS.map((x) => (pseudoWeather === x ? 1 : 0)))
  f.push(...WEATHERS.map((x) => (weather === x ? 1 : 0)))
  f.push(
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

  f.push(...([true, "copyvolatile", "shedtail"] as const).map((x) => (selfSwitch === x ? 1 : 0)))
  f.push(...["Wish", "healingwish", "revivalblessing"].map((x) => (slotCondition === x ? 1 : 0)))
  f.push(
    ...["def", "target"].map((x) =>
      overrideDefensiveStat ?? overrideDefensivePokemon === x ? 1 : 0
    )
  )
  f.push(
    ...["def", "target"].map((x) =>
      (overrideOffensiveStat ?? overrideOffensivePokemon) === x ? 1 : 0
    )
  )

  const effect = reconcileEffect(move)
  f.push(...encodeMoveEffect(effect))

  return f
}
