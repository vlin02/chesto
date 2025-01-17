import { Generations } from "@pkmn/data";
import { Dex } from "@pkmn/dex";

const gen = new Generations(Dex).get(9)
console.log(gen.moves.get("leech seed"))