import { MOVE_CATEGORIES, TYPE_NAMES } from "../../battle.js"
import { Move } from "@pkmn/data"

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

  const x: number[] = []

  x.push(basePower / 100)
  x.push(...MOVE_CATEGORIES.map((x) => (category === x ? 1 : 0)))
  x.push((accuracy === true ? 100 : accuracy) / 100)
  x.push(priority)
  x.push(...TYPE_NAMES.map((x) => (type === x ? 1 : 0)))
  
  x.push(pp / 10)
  x.push(...MOVE_FLAGS.map((k) => (flags[k] ? 1 : 0)))
  x.push(drain ? drain[0] / drain[1] : 0)
  x.push(willCrit ? 1 : [1 / 24, 1 / 8, 1 / 2, 1][critRatio!])

  x.push(heal ? heal[0] / heal[1] : 0)
  x.push(recoil ? recoil[0] / recoil[1] : 0)

  x.push(...["always", "ifhit"].map((x) => (selfdestruct === x ? 1 : 0)))

  x.push(...SIDE_CONDITIONS.map((x) => (sideCondition === x ? 1 : 0)))
  x.push(...(Array.isArray(multihit) ? multihit : [multihit ?? 0, multihit ?? 0]))

  x.push(duration ?? 0)

  x.push(...["level"].map((x) => (damage === x ? 1 : 0)))

  x.push(
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

  x.push(...TERRAINS.map((x) => (terrain === x ? 1 : 0)))
  x.push(...PSEUDO_WEATHERS.map((x) => (pseudoWeather === x ? 1 : 0)))
  x.push(...WEATHERS.map((x) => (weather === x ? 1 : 0)))

  x.push(...([true, "copyvolatile", "shedtail"] as const).map((x) => (selfSwitch === x ? 1 : 0)))
  x.push(...["Wish", "healingwish", "revivalblessing"].map((x) => (slotCondition === x ? 1 : 0)))
  x.push(
    ...["def", "target"].map((x) =>
      overrideDefensiveStat ?? overrideDefensivePokemon === x ? 1 : 0
    )
  )
  x.push(
    ...["def", "target"].map((x) =>
      (overrideOffensiveStat ?? overrideOffensivePokemon) === x ? 1 : 0
    )
  )

  // const effect = reconcileEffect(move)
  // x.push(...encodeMoveEffect(effect))

  return x
}
