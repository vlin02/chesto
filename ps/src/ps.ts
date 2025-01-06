import { toID } from "@pkmn/data"
import { Request, Protocol } from "@pkmn/protocol"

export function parseRequest(request: any) {
  if (!request.requestType) {
    request.requestType = "move"
    if (request.forceSwitch) {
      request.requestType = "switch"
    } else if (request.teamPreview) {
      request.requestType = "team"
    } else if (request.wait) {
      request.requestType = "wait"
    }
  }

  if (request.requestType === "wait") request.noCancel = true
  if (request.side) {
    for (const pokemon of request.side.pokemon) {
      Protocol.parseDetails(pokemon.ident.substr(4), pokemon.ident, pokemon.details, pokemon)
      Protocol.parseDetails(pokemon.condition, pokemon)
      pokemon.ability = pokemon.ability || pokemon.baseAbility
    }
  }

  if (request.active) {
    request.active = request.active.map((active: any, i: number) =>
      request.side.pokemon[i].fainted ? null : active
    )
    for (const active of request.active) {
      if (!active) continue
      for (const move of active.moves) {
        if (move.move) move.name = move.move
        move.id = toID(move.name)
      }
      if (active.maxMoves) {
        if (active.maxMoves.maxMoves) {
          active.canGigantamax = active.maxMoves.gigantamax
          active.maxMoves = active.maxMoves.maxMoves
        }
        for (const move of active.maxMoves) {
          move.id = move.move
        }
      }
      if (active.canZMove) {
        active.zMoves = active.canZMove
        for (const move of active.zMoves) {
          if (!move) continue
          if (move.move) move.name = move.move
          move.id = toID(move.name)
        }
      }
    }
  }

  return request as Request
}
