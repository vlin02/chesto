import { Generation } from "@pkmn/data"
import { Patch } from "./version.js"
import { Observer } from "./client/observer.js"

export type Format = {
  gen: Generation
  patch: Patch
}

export type Run = {
  fmt: Format
  obs: Observer
}
