import { createReadStream } from "fs"
import { createInterface } from "readline"

const fileStream = createReadStream("replays-2.jsonl")
const rl = createInterface({
  input: fileStream,
  crlfDelay: Infinity
})

let counts = new Map<string, number>()
let i = 0
for await (const line of rl) {
  i += 1

  const { inputlog }: { inputlog: string } = JSON.parse(line)

  for (const input of inputlog.split("\n")) {
    const tokens = input.split(" ")
    const pfx = [">p1", ">p2"].includes(tokens[0]) ? tokens.slice(0, 2).join(" ") : tokens[0]
    counts.set(pfx, (counts.get(pfx) ?? 0) + 1)
  }
}
console.log(counts)