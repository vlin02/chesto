import { Battle } from "@pkmn/client"

export function parse(battle: Battle) {
  const { p1, p2 } = battle

  for (const {
    species,
    hp,
    maxhp,
    level,
    gender,
    shiny,
    baseMaxhp,
    fainted,
    status,
    teraType,
    terastallized,
    canTerastallize,
    trapped,
    timesAttacked,
    hurtThisTurn,
    volatiles,
    hpcolor,
    newlySwitched,
    beingCalledBack,
    statusStage,
    statusState,
    boosts,
    ability,
    baseAbility,
    illusion,
    revealedDetails,
    item,
    itemEffect,
    lastItem,
    lastItemEffect,
    teamPreviewItem,
    name,
    nature,
    hpType,
    moves,
    weighthg,
    speciesForme,
    position,
    evs,
    ivs,
    moveSlots,
    moveThisTurn,
    movesUsedWhileActive,
    maxMoves,
    zMoves,
    canDynamax,
    canGigantamax,
    canUltraBurst,
    canMegaEvo
  } of p1.team) {
    /*
    stats direct calculation
    */


    return {
      species_num: species.num,
      hp,
      maxhp,
      level,
      fainted,
      status,
      statusStage,
      statusState,
    }
  }
}
