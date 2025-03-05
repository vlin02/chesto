import WebSocket from "ws"
import { piped } from "../parse.js"
import { Observer } from "../parser/observer.js"
import { Generations } from "@pkmn/data"
import { Dex } from "@pkmn/dex"
import { Run } from "../run.js"
import { randomAgent, chioceToMessage as choiceToMessage } from "./agents.js"

const WS_URL = "ws://localhost:8001/showdown/websocket"
const SERVER_URL = "https://play.pokemonshowdown.com/"

async function assertAnon(name: string, challstr: string) {
  let res = await fetch(
    new URL(`api/getassertion?userid=${name}&challstr=${challstr}`, SERVER_URL),
    {
      method: "GET"
    }
  )

  return await res.text()
}

type Event =
  | {
      type: "message"
      roomId: string
      logs: string[]
    }
  | {
      type: "request"
      username: string
      format: string
    }

class Session {
  challstr!: string
  username!: string
  ws: WebSocket
  loggingIn?: () => void
  challenging?:
    | { status: "pending"; next: (v: Promise<string | null>) => void }
    | {
        status: "waiting"
        next: (id: string | null) => void
      }

  on: (event: Event) => void

  constructor(ws: WebSocket) {
    this.ws = ws
    this.on = () => {}
  }

  start() {
    let onReady

    this.ws.on("message", async (data: WebSocket.RawData) => {
      const msg = data.toString()

      let p = piped(msg, 1)

      if (msg.startsWith(">")) {
        const [roomId, ...logs] = msg.slice(1).split("\n")
        this.on({
          type: "message",
          roomId,
          logs
        })
      }

      switch (p.args[0]) {
        case "challstr":
          onReady!()
          this.challstr = msg.slice(p.i)
          break
        case "updateuser":
          p = piped(msg, p.i)

          this.loggingIn?.()
          this.username = p.args[0]

          break
        case "updatechallenges":
          break
        case "pm":
          p = piped(msg, p.i, 3)

          if (
            p.args[0] === this.username &&
            p.args[2].startsWith("/challenge") &&
            !p.args[2].endsWith("/challenge")
          ) {
            if (this.challenging?.status !== "pending") throw ""
            let done
            const v = new Promise<string | null>((res) => {
              done = res
            })
            this.challenging!.next(v)
            this.challenging = {
              status: "waiting",
              next: done!
            }
          }

          if (p.args[1] == this.username) {
            if (p.args[2].includes("rejected the challenge")) {
              if (this.challenging?.status !== "waiting") throw ""
              this.challenging.next(null)
              delete this.challenging
            } else if (p.args[2].includes("accepted the challenge")) {
              if (this.challenging?.status !== "waiting") throw ""
              const i = p.args[2].indexOf(`href="/`) + 7
              const j = p.args[2].indexOf(`"`, i)
              this.challenging.next(p.args[2].slice(i, j))
              delete this.challenging
            }
          }
          break
        case "updatesearch":
          break
      }
    })

    return new Promise<void>((res) => {
      onReady = res
    })
  }

  login(name: string, assertion: string) {
    this.ws.send(`|/trn ${name},0,${assertion}`)
    return new Promise<void>((res) => {
      this.loggingIn = res
    })
  }

  challenge(name: string, format: string, team = null) {
    this.ws.send(`|/utm ${team ? team : "null"}`)
    this.ws.send(`|/challenge ${name}, ${format}`)
    return new Promise<() => Promise<string | null>>((res) => {
      this.challenging = { status: "pending", next: (x) => res(() => x) }
    })
  }

  accept(name: string, team = null) {
    this.ws.send(`|/utm ${team ? team : "null"}`)
    this.ws.send(`|/accept ${name}`)
  }

  reject(name: string) {
    this.ws.send(`|/reject ${name}`)
  }

  send(roomId: string, msg: string) {
    this.ws.send(`${roomId}|${msg}`)
  }
}

const s1 = new Session(new WebSocket(WS_URL))
await s1.start()
await s1.login("chest20", await assertAnon("chest20", s1.challstr))

// const s2 = new Session(new WebSocket(WS_URL))
// await s2.start()
// await s2.login("chest21", await assertAnon("chest21", s2.challstr))

const find = await s1.challenge("chest17", "gen9randombattle")

const id = await find()
const gen = new Generations(Dex).get(9)
const obs = new Observer(gen)

const run: Run = {
  obs,
  gen
}

let pendingReq = false

s1.on((event) => {
  
})

// .set(id!, (logs) => {
//   console.log(logs)
//   for (const log of logs) {
//     obs.read(log)
//   }

//   if (pendingReq) {
//     const choice = randomAgent(run)
//     s1.send(id!, "/" + choiceToMessage(choice))
//     pendingReq = false
//   }

//   pendingReq = logs[0].startsWith("|request")
// })
