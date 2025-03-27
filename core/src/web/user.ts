import WebSocket from "ws"
import { piped } from "../parse.js"
import assert from "assert"
import { getAnonAssertion } from "./api.js"
import { EventEmitter } from "node:events"

type LogingRequest = {
  next: () => void
}

type ChallengeRequest =
  | { status: "inflight"; next: (p: Promise<string | null>) => void }
  | { status: "pending"; next: (id: string | null) => void }

type SearchRequest =
  | { status: "inflight"; next: (p: Promise<void>) => void }
  | { status: "pending"; next: () => void }

function extractBattleId(s: string) {
  const match = s.match(/href="\/([^"]+)"/)
  return match?.[1]
}

export class User {
  challstr!: string
  username!: string
  ws: WebSocket
  private loginReq?: LogingRequest
  private challengeReq?: ChallengeRequest
  private searchReqs: Map<string, SearchRequest>

  global: EventEmitter<{
    search: [string[]]
  }>

  room: EventEmitter<{ [k: string]: [string[]] }>

  constructor(ws: WebSocket) {
    this.ws = ws
    this.global = new EventEmitter()
    this.room = new EventEmitter()
    this.searchReqs = new Map()
  }

  start() {
    return new Promise<void>((res) => {
      this.ws.on("message", (data: WebSocket.RawData) => {
        const msg = data.toString()
        let p = piped(msg, 1)

        if (msg.startsWith(">")) {
          const [roomId, ...logs] = msg.slice(1).split("\n")
          this.room.emit(roomId, logs)
        }

        switch (p.args[0]) {
          case "challstr":
            this.challstr = msg.slice(p.i)
            res()
            break

          case "updateuser":
            p = piped(msg, p.i)
            this.loginReq?.next()
            this.username = p.args[0]
            break

          case "pm":
            p = piped(msg, p.i, 3)
            const [sender, receiver, pm] = p.args

            if (
              sender === this.username &&
              pm.startsWith("/challenge") &&
              !pm.endsWith("/challenge")
            ) {
              assert(this.challengeReq?.status === "inflight")

              let next: (id: string | null) => void
              const invite = new Promise<string | null>((res) => {
                next = res
              })
              this.challengeReq.next(invite)
              this.challengeReq = { status: "pending", next: next! }
            }

            if (receiver === this.username) {
              if (pm.includes("rejected the challenge")) {
                assert(this.challengeReq?.status === "pending")
                this.challengeReq.next(null)
                delete this.challengeReq
              } else if (pm.includes("accepted the challenge")) {
                assert(this.challengeReq?.status === "pending")
                this.challengeReq.next(extractBattleId(pm)!)
                delete this.challengeReq
              }
            }
            break

          case "updatesearch":
            const { searching, games } = JSON.parse(msg.slice(p.i)) as {
              searching: string[]
              games: null | { [k: string]: string }
            }
            this.global.emit("search", games ? [...Object.keys(games)] : [])

            for (const [formatId, req] of this.searchReqs) {
              if (searching.includes(formatId) && req.status === "inflight") {
                const p = new Promise<void>((res) => {
                  this.searchReqs.set(formatId, { status: "pending", next: res })
                })
                req.next(p)
              }

              if (!searching.includes(formatId) && req.status === "pending") {
                req.next()
                this.searchReqs.delete(formatId)
              }
            }

            break
        }
      })
    })
  }

  static async anon(ws: WebSocket, name: string) {
    const u = new User(ws)
    await u.start()
    await u.trn(name, await getAnonAssertion(name, u.challstr))
    return u
  }

  trn(name: string, assertion: string) {
    this.ws.send(`|/trn ${name},0,${assertion}`)
    return new Promise<void>((res) => {
      this.loginReq = { next: res }
    })
  }

  challenge(name: string, format: string, team = null) {
    this.ws.send(`|/utm ${team ?? "null"}`)
    this.ws.send(`|/challenge ${name}, ${format}`)
    return new Promise<() => Promise<string | null>>((res) => {
      this.challengeReq = { status: "inflight", next: (x) => res(() => x) }
    })
  }

  accept(name: string, team = null) {
    this.ws.send(`|/utm ${team ?? "null"}`)
    this.ws.send(`|/accept ${name}`)
  }

  reject(name: string) {
    this.ws.send(`|/reject ${name}`)
  }

  send(roomId: string, msg: string) {
    this.ws.send(`${roomId}|${msg}`)
  }

  cancelSearch() {
    this.ws.send(`|/cancelSearch`)
  }

  search(formatId: string) {
    const p = new Promise<() => Promise<void>>((res) => {
      this.searchReqs.set(formatId, { status: "inflight", next: (x) => res(() => x) })
    })
    this.ws.send(`|/utm null`)
    this.ws.send(`|/search ${formatId}`)
    return p
  }
}

export async function loginAnon(ws: WebSocket, name: string) {
  const u = new User(ws)
  await u.start()
  await u.trn(name, await getAnonAssertion(name, u.challstr))
  return u
}
