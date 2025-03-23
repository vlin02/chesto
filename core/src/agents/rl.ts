import { Observer } from "../parser/observer.js"
import { Pool } from "undici"
import { encodeBattle } from "../env/model/state.js"
import { BSON } from "bson"
import { packBinary } from "../env/transport.js"


export async function predict(pool: Pool, observers: Observer[], path: string) {
  let { body } = await pool.request({
    path: "/predict",
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: BSON.serialize({
      path,
      states: observers.map((obs) => packBinary(encodeBattle(obs)))
    })
  })

  return body.json()
}