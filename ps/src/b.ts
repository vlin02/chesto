import { Generations } from "@pkmn/data";
import { Dex } from "@pkmn/dex";
import { isPressured } from "./parse/move.js";

const gen = new Generations(Dex).get(9)
