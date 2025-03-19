const SERVER_URL = "https://play.pokemonshowdown.com/"

export async function getAnonAssertion(name: string, challstr: string) {
  let res = await fetch(
    new URL(`api/getassertion?userid=${name}&challstr=${challstr}`, SERVER_URL),
    {
      method: "GET"
    }
  )

  return await res.text()
}
