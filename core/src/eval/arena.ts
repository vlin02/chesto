import { getAnonAssertion } from "../web/binding.js"
import { User } from "../web/user.js"
import { WebSocket } from "ws"

export async function login(url: string, name: string) {
  const u = new User(new WebSocket(url))
  await u.start()
  await u.trn(name, await getAnonAssertion(name, u.challstr))
  return u
}

export class BattlePool {
  u1: User
  u2: User

  private p: Promise<any>

  constructor(u1: User, u2: User) {
    this.u1 = u1
    this.u2 = u2
    this.p = Promise.resolve()
  }

  async start() {
    return (this.p = new Promise<string>(async (resolve) => {
      await this.p
      const find = await this.u1.challenge(this.u2.username, "gen9randombattle")
      this.u2.accept(this.u1.username)
      resolve((await find())!)
    }))
  }
}

