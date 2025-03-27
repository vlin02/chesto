import { Pool } from "undici"
import { BSON } from "bson"

export type Prediction = {
  probs: number[]
  action_id: number
  value: number
}

export async function predict(pool: Pool, modelPath: string, states: any[]) {
  let { body } = await pool.request({
    path: "/predict",
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: BSON.serialize({
      modelPath,
      states
    })
  })

  return body.json() as any as Prediction[]
}
