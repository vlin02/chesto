import { format } from "url"
import { Pool } from "undici"
import { createWriteStream } from "fs"
import stream from "stream/promises"

const api = new Pool("https://pokemonshowdown.com", {
  connections: 10,
  pipelining: 10
})

const replayApi = new Pool("https://replay.pokemonshowdown.com/", {
  connections: 2000,
  pipelining: 50
})

export type Replay = {
  uploadtime: number
  players: [string, string]
  rating: number
  id: string
  inputlog: string
}

export async function fetchReplays(formatId: string) {
  let { body } = await api.request({
    path: `/ladder/${formatId}.json`,
    method: "GET"
  })

  const { toplist } = (await body.json()) as { toplist: [{ userid: string }] }

  const allReplays: Record<string, Replay> = {}

  let userIds = toplist.map((x) => x.userid)
  const allUserIds = new Set(userIds)

  while (userIds.length) {
    let replays = await Promise.all(
      userIds.map(async (id) => {
        let before: number | null = null
        const replays = []

        while (true) {
          try {
            let { body } = await replayApi.request({
              method: "GET",
              path: format({
                pathname: `/api/replays/search.json`,
                query: {
                  user: id,
                  format: formatId,
                  before
                }
              })
            })
            let page = (await body.json()) as {
              uploadtime: number
              players: [string, string]
              rating: number
              id: string
            }[]

            replays.push(...page)

            if (page.length < 51) break

            before = page[50].uploadtime
          } catch (e) {
            console.log(e)
          }
        }

        return replays
      })
    ).then((x) => x.flat())

    userIds = []

    for (const replay of replays.filter((x) => x.rating > 1500)) {
      const { players } = replay
      for (const player of players) {
        let userId = player.toLowerCase()

        if (allUserIds.has(userId)) continue
        allUserIds.add(userId)
        userIds.push(userId)
      }
    }

    for (const replay of replays.filter((x) => x.rating > 1600)) {
      allReplays[replay.id] = replay
    }

    console.log(Object.keys(allReplays).length, userIds.length)
  }

  return allReplays
}

const replays = await fetchReplays("gen9randombattle")

let i = 0

const allJson = await Promise.all(
  Object.values(replays).map(async (x) => {
    while (true) {
      try {
        let { body } = await replayApi.request({
          method: "GET",
          path: `/${x.id}.inputlog`
        })
        i += 1
        console.log(i)

        return { ...x, inputlog: await body.text() }
      } catch (e) {
        console.log(e)
      }
    }
  })
)

const writeStream = createWriteStream("replays-1.jsonl")
for (const row of allJson.sort((a, b) => a.uploadtime - b.uploadtime)) {
  writeStream.write(JSON.stringify(row) + "\n")
}
writeStream.end()
await stream.finished(writeStream)

// writeFileSync(JSON.stringify(await fetchReplays("gen9randombattle")), "all.json")
