import { format } from "url"
import { Observation } from "../features/observation.js"
import { Options } from "../features/options.js"

type Input = {
  observation: Observation
  options: Options
}

export async function rlAgent(input: Input) {
  return fetch(
    format({
      protocol: "http",
      hostname: "localhost",
      port: 5000,
      pathname: "/predict"
    }),
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(input)
    }
  )
}
