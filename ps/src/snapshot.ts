import { Battle } from "@pkmn/client"

function extractBattle(battle: Battle) {
  const { p1, p2 } = battle

  for (const {
    // CONSIDERED
    // does opp team poke have set ?
    set,

    level,
    gender,
    terastallized,
    hp,
    maxhp,
    baseMaxhp,
    status,
    fainted,
    boosts,

    // find cases where these are true ?
    maybeTrapped,
    maybeDisabled,

    statusStage,
    statusState,
    volatiles,

    item,
    // is this any different than set nature ?
    canTerastallize,
    trapped,
    moveSlots,

    moveThisTurn,
    hurtThisTurn,

    weighthg,
    types,
    addedType,
    switching,

    isGrounded,
    effectiveAbility,
    isActive,

    speciesForme,

    // NOT BEING CONSIDERED

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

    // from set
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

    // derived from terastallize
    teraType,
    isTerastallized,

    // derived from moveslots
    moves,

    //derived from this.item
    hasItem
  } of p1.team) {
    return {}
  }
}
