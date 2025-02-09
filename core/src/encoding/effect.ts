import { StatusName, Move } from "@pkmn/data"
import { MOD_STAT_IDS, ModStatId, STATUS_IDS, StatusId } from "../battle.js"

const VOLATILE_STATUSES = [
  "flinch",
  "aquaring",
  "attract",
  "confusion",
  "banefulbunker",
  "partiallytrapped",
  "mustrecharge",
  "burningbulwark",
  "charge",
  "curse",
  "defensecurl",
  "destinybond",
  "protect",
  "disable",
  "dragoncheer",
  "encore",
  "endure",
  "focusenergy",
  "followme",
  "gastroacid",
  "glaiverush",
  "helpinghand",
  "imprison",
  "ingrain",
  "leechseed",
  "magnetrise",
  "minimize",
  "noretreat",
  "lockedmove",
  "powertrick",
  "healblock",
  "ragepowder",
  "roost",
  "saltcure",
  "substitute",
  "silktrap",
  "smackdown",
  "sparklingaria",
  "spikyshield",
  "stockpile",
  "syrupbomb",
  "tarshot",
  "taunt",
  "torment",
  "uproar",
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
    chance = 100,
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
  const p = chance / 100
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

function encodeUserEffect({ boosts, status, volatileStatus }: UserEffect) {
  const f: number[] = []

  f.push(
    ...MOD_STAT_IDS.flatMap((id) => {
      const { n, p } = boosts[id] ?? { n: 0, p: 0 }
      return [n, p]
    })
  )

  f.push(
    ...STATUS_IDS.map((id) => {
      return status?.id === id ? status.p : 0
    })
  )

  f.push(
    ...VOLATILE_STATUSES.map((s) => {
      return s === volatileStatus?.name ? volatileStatus.p : 0
    })
  )

  return f
}

export function encodeMoveEffect({ self, opp }: MoveEffect) {
  return [...encodeUserEffect(self), ...encodeUserEffect(opp)]
}
