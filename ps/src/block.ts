export class BlockReader {
  private leftover: Buffer = Buffer.alloc(0)

  load(chunk: Buffer): Buffer[] {
    const seq = Buffer.concat([this.leftover, chunk])
    const blocks: Buffer[] = []
    let i = 0

    while (i + 4 <= seq.length) {
      const length = seq.readUInt32BE(i)
      if (seq.length < i + 4 + length) break
      blocks.push(seq.subarray(i + 4, i + 4 + length))
      i += 4 + length
    }

    this.leftover = seq.subarray(i)
    return blocks
  }

  static prefix(num: number): Buffer {
    const buffer = Buffer.alloc(4)
    buffer.writeUInt32BE(num)
    return buffer
  }
}
