import { Generation } from "@pkmn/data"
import { Observer } from "../parser/observer.js"
import { Choice } from "../parser/option.js"
import { User } from "../parser/user.js"

// Constants from original implementation
const ENTRY_HAZARDS = {
  "Spikes": "Spikes",
  "Stealth Rock": "Stealth Rock",
  "Sticky Web": "Sticky Web",
  "Toxic Spikes": "Toxic Spikes"
}

const ANTI_HAZARD_MOVES = ["Rapid Spin", "Defog"]

const SPEED_TIER_COEFFICIENT = 0.1
const HP_FRACTION_COEFFICIENT = 0.4
const SWITCH_OUT_MATCHUP_THRESHOLD = -2

// Helper function to estimate matchup similar to _estimate_matchup
function estimateMatchup(gen: Generation, ally: User, foe: User): number {
  // Get max damage multiplier ally types have against foe
  const allyOffense = Math.max(
    ...ally.types.map((t) => {
      if (!gen.types.get(t)) return 1
      return Math.max(...foe.defensiveTyping.map((dt) => gen.types.get(t)!.effectiveness[dt] || 1))
    })
  )

  // Get max damage multiplier foe types have against ally
  const foeOffense = Math.max(
    ...foe.types.map((t) => {
      if (!gen.types.get(t)) return 1
      return Math.max(...ally.defensiveTyping.map((dt) => gen.types.get(t)!.effectiveness[dt] || 1))
    })
  )

  let score = allyOffense - foeOffense

  // Speed tier advantage
  const allySpeed = ally.stats?.spe || 0
  const foeSpeed = foe.stats?.spe || 0
  if (allySpeed > foeSpeed) {
    score += SPEED_TIER_COEFFICIENT
  } else if (foeSpeed > allySpeed) {
    score -= SPEED_TIER_COEFFICIENT
  }

  // HP fraction
  const allyHpFraction = ally.hp[0] / ally.hp[1]
  const foeHpFraction = foe.hp[0] / foe.hp[1]
  score += allyHpFraction * HP_FRACTION_COEFFICIENT
  score -= foeHpFraction * HP_FRACTION_COEFFICIENT

  return score
}

// Helper function to decide if we should switch out
function shouldSwitchOut(obs: Observer, ally: User, foe: User): boolean {
  // Check if there are decent switches available
  const goodSwitches = obs.listSwitches().filter((species) => {
    const switchMon = obs.ally.team[species]
    return estimateMatchup(obs.gen, switchMon, foe) > 0
  })

  if (goodSwitches.length > 0) {
    // Check if there's a good reason to switch
    const { boosts } = ally
    if ((boosts.def || 0) <= -3 || (boosts.spd || 0) <= -3) {
      return true
    }

    if ((boosts.atk || 0) <= -3 && (ally.stats?.atk || 0) >= (ally.stats?.spa || 0)) {
      return true
    }

    if ((boosts.spa || 0) <= -3 && (ally.stats?.atk || 0) <= (ally.stats?.spa || 0)) {
      return true
    }

    if (estimateMatchup(obs.gen, ally, foe) < SWITCH_OUT_MATCHUP_THRESHOLD) {
      return true
    }
  }

  return false
}

// Helper function to estimate effective stat values with boosts
function statEstimation(mon: User, stat: "atk" | "def" | "spa" | "spd" | "spe"): number {
  const baseStat = mon.stats?.[stat] || 80 // fallback value
  const boost = mon.boosts[stat] || 0

  // Apply boost calculation similar to the Python implementation
  let boostMultiplier = 1
  if (boost > 0) {
    boostMultiplier = (2 + boost) / 2
  } else if (boost < 0) {
    boostMultiplier = 2 / (2 - boost)
  }

  return baseStat * boostMultiplier
}

// Main heuristic function
export function simpleHeuristic(obs: Observer): Choice {
  const option = obs.getOption()
  if (!option) return { type: "move", move: "Struggle", tera: false }

  const { select, switches } = option
  const ally = obs.ally.active
  const foe = obs.foe.active

  // See if we should switch out and have available switches
  if (shouldSwitchOut(obs, ally, foe) && switches.length > 0) {
    // Find best switch-in
    const bestSwitch = switches.reduce((best, species) => {
      const mon = obs.ally.team[species]
      const score = estimateMatchup(obs.gen, mon, foe)
      return score > estimateMatchup(obs.gen, obs.ally.team[best], foe) ? species : best
    }, switches[0])

    return { type: "switch", species: bestSwitch }
  }

  // If we have moves available
  if (select) {
    const moves =
      select.type === "default"
        ? select.moves
        : [select.type === "struggle" ? "Struggle" : "Recharge"]
    const canTera = select.type === "default" && select.tera

    // Count remaining Pokémon
    const remainingMons = Object.values(obs.ally.team).filter((m) => m.hp[0] > 0).length
    const remainingFoes = Object.values(obs.foe.team).filter((m) => m.hp[0] > 0).length || 6 // Assume 6 if unknown

    // Check for entry hazard setup opportunity
    if (remainingFoes >= 3) {
      for (const move of moves) {
        if (Object.keys(ENTRY_HAZARDS).includes(move)) {
          const hazardName = ENTRY_HAZARDS[move as keyof typeof ENTRY_HAZARDS]
          if (!(hazardName in obs.foe.effects)) {
            return { type: "move", move, tera: false }
          }
        }
      }
    }

    // Check for hazard removal
    if (Object.keys(obs.ally.effects).length > 0 && remainingMons >= 2) {
      for (const move of moves) {
        if (ANTI_HAZARD_MOVES.includes(move)) {
          return { type: "move", move, tera: false }
        }
      }
    }

    // Check for setup moves
    if (ally.hp[0] === ally.hp[1] && estimateMatchup(obs.gen, ally, foe) > 0) {
      for (const move of moves) {
        const moveData = obs.gen.moves.get(move)
        if (
          moveData?.self?.boosts &&
          Object.values(moveData.self.boosts).reduce((sum, v) => sum + v, 0) >= 2
        ) {
          // Check if we haven't maxed boosts already
          const canBoostMore = Object.entries(moveData.self.boosts).some(
            ([stat, val]) => val > 0 && (ally.boosts[stat as keyof typeof ally.boosts] || 0) < 6
          )

          if (canBoostMore) {
            return { type: "move", move, tera: false }
          }
        }
      }
    }

    // Choose best damaging move
    const physicalRatio = statEstimation(ally, "atk") / statEstimation(foe, "def")
    const specialRatio = statEstimation(ally, "spa") / statEstimation(foe, "spd")

    let bestMove = moves[0]
    let bestScore = -Infinity

    for (const move of moves) {
      const moveData = obs.gen.moves.get(move)
      if (!moveData) continue

      const basePower = moveData.basePower
      const stab = ally.types.includes(moveData.type) ? 1.5 : 1
      const damageRatio = moveData.category === "Physical" ? physicalRatio : specialRatio
      const accuracy = moveData.accuracy === true ? 1 : moveData.accuracy / 100
      const expectedHits = moveData.multihit
        ? Array.isArray(moveData.multihit)
          ? (moveData.multihit[0] + moveData.multihit[1]) / 2
          : moveData.multihit
        : 1

      // Calculate effectiveness
      const typeEffectiveness = foe.defensiveTyping.reduce(
        (eff, type) => eff * (obs.gen.types.get(moveData.type)?.effectiveness[type] || 1),
        1
      )

      const moveScore = basePower * stab * damageRatio * accuracy * expectedHits * typeEffectiveness

      if (moveScore > bestScore) {
        bestScore = moveScore
        bestMove = move
      }
    }

    // Use Terastallize for our best move if available
    return { type: "move", move: bestMove, tera: canTera }
  }

  // Fallback to switching if we can't move
  if (switches.length > 0) {
    const bestSwitch = switches.reduce((best, species) => {
      const mon = obs.ally.team[species]
      const score = estimateMatchup(obs.gen, mon, foe)
      return score > estimateMatchup(obs.gen, obs.ally.team[best], foe) ? species : best
    }, switches[0])

    return { type: "switch", species: bestSwitch }
  }

  // Ultimate fallback
  return { type: "move", move: "Struggle", tera: false }
}
