import { Generations } from "@pkmn/data"
import { Dex } from "@pkmn/dex"
import { extractMove } from "../snapshot.js"
import { assert } from "console"

const gen = new Generations(Dex).get(9)
const EPS = 0.01

function eq(a: number, b: number) {
  return Math.abs(a - b) < EPS
}

{
  const { multiHit } = extractMove(gen.moves.get("Triple Axel")!)
  const {
    times: [min, max],
    recheckAcc
  } = multiHit

  assert(min === 3)
  assert(max === 3)
  assert(recheckAcc === true)
}

{
  const { heal } = extractMove(gen.moves.get("Slack Off")!)
  assert(heal === 0.5)
}

{
  const {
    statOverride: { offensive }
  } = extractMove(gen.moves.get("bodypress")!)
  assert(offensive === "def")
}

{
  const {
    statOverride: { defensive }
  } = extractMove(gen.moves.get("Psyshock")!)
  assert(defensive === "def")
}

{
  const { thawsTarget } = extractMove(gen.moves.get("Scorching Sands")!)
  assert(thawsTarget === true)
}
{
  const { selfSwitch } = extractMove(gen.moves.get("Volt Switch")!)
  assert(selfSwitch === true)
}
{
  const { selfDestruct } = extractMove(gen.moves.get("Explosion")!)
  assert(selfDestruct === true)
}

{
  const { recoil } = extractMove(gen.moves.get("Flare Blitz")!)
  assert(eq(recoil, 1 / 3), `${recoil}`)
}

{
  const { effects } = extractMove(gen.moves.get("sludgebomb")!)
  const [status, p] = effects.foe.status!
  assert(status === "psn")
  assert(p && eq(p, 0.3))
}

{
  const { drain } = extractMove(gen.moves.get("gigadrain")!)
  assert(eq(drain, 0.5))
}

{
  const { effects } = extractMove(gen.moves.get("outrage")!)
  const [volatile, p] = effects.ally.volatile!
  assert(volatile === "lockedmove", volatile)
  assert(eq(p, 1))
}

{
  const {
    ignore: { evasion, defensive }
  } = extractMove(gen.moves.get("sacredsword")!)
  assert(defensive === true)
  assert(evasion === true)
}

{
  const { breaksProtect } = extractMove(gen.moves.get("hyperspacefury")!)
  assert(breaksProtect === true)
}

{
  const {
    ignore: { ability }
  } = extractMove(gen.moves.get("sunsteelstrike")!)
  assert(ability === true)
}

{
  const { willCrit } = extractMove(gen.moves.get("wickedblow")!)
  assert(willCrit === true)
}

{
  const { noPPBoosts } = extractMove(gen.moves.get("revivalblessing")!)
  assert(noPPBoosts === true)
}

{
  const { effects } = extractMove(gen.moves.get("toxicspikes")!)
  assert(effects.foe.sideCondition === "toxicspikes")
}

{
  const { effects } = extractMove(gen.moves.get("lightscreen")!)
  assert(effects.ally.sideCondition === "lightscreen")
}

{
  const { effects } = extractMove(gen.moves.get("protect")!)
  const [volatile, p] = effects.ally.volatile!
  assert(volatile === "protect")
  assert(p === 1)
}

{
  const { weather } = extractMove(gen.moves.get("raindance")!)
  assert(weather === "Rain")
}

{
  const { effects } = extractMove(gen.moves.get("firefang")!)
  const { status: [status, p1] = [], volatile: [volatile, p2] = [] } = effects.foe
  assert(status === "brn")
  assert(eq(p1!, 0.1))
  assert(volatile === "flinch")
  assert(eq(p2!, 0.1))
}
