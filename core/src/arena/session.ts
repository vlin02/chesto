import WebSocket from "ws"
import { piped } from "../parse.js"
import assert from "assert"

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

type LogingRequest = {
  next: () => void
}

type ChallengeRequest =
  | { status: "inflight"; next: (p: Promise<string | null>) => void }
  | { status: "pending"; next: (id: string | null) => void }

function extractBattleId(s: string) {
  const match = s.match(/href="\/([^"]+)"/)
  return match?.[1]
}

export class Session {
  challstr!: string
  username!: string
  ws: WebSocket
  loginReq?: LogingRequest
  challengeReq?: ChallengeRequest
  on: (event: Event) => void = () => {}

  constructor(ws: WebSocket) {
    this.ws = ws
  }

  start() {
    return new Promise<void>((res) => {
      this.ws.on("message", (data: WebSocket.RawData) => {
        const msg = data.toString()
        let p = piped(msg, 1)

        if (msg.startsWith(">")) {
          const [roomId, ...logs] = msg.slice(1).split("\n")
          this.on({ type: "message", roomId, logs })
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

          case "updatechallenges":
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
            break
        }
      })
    })
  }

  login(name: string, assertion: string) {
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
}
