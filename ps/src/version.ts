import { Battle as Battle_0_8 } from "@pkmn/sim-0.8.9"
import { Battle as Battle_0_9 } from "@pkmn/sim-0.9.26"

export type AnyBattle = Battle_0_8 | Battle_0_9
export type AnyBattleT = typeof Battle_0_8 | typeof Battle_0_9

type Release = {
  timestamp: string
  v: string
}

const RELEASES: Release[] = [
  { timestamp: "2023-01-03T21:24:28.389Z", v: "0.7.13" },
  { timestamp: "2023-01-07T21:15:04.470Z", v: "0.7.14" },
  { timestamp: "2023-01-09T20:37:04.417Z", v: "0.7.15" },
  { timestamp: "2023-01-20T03:16:20.080Z", v: "0.7.16" },
  { timestamp: "2023-01-21T01:37:38.914Z", v: "0.7.17" },
  { timestamp: "2023-01-22T03:16:23.113Z", v: "0.7.18" },
  { timestamp: "2023-01-29T19:35:07.841Z", v: "0.7.19" },
  { timestamp: "2023-01-29T21:33:54.921Z", v: "0.7.20" },
  { timestamp: "2023-02-02T00:05:24.933Z", v: "0.7.21" },
  { timestamp: "2023-02-13T19:06:20.842Z", v: "0.7.22" },
  { timestamp: "2023-02-24T03:30:22.915Z", v: "0.7.23" },
  { timestamp: "2023-02-27T00:10:46.032Z", v: "0.7.24" },
  { timestamp: "2023-02-28T16:34:19.064Z", v: "0.7.25" },
  { timestamp: "2023-03-01T23:03:58.373Z", v: "0.7.26" },
  { timestamp: "2023-03-09T23:25:35.423Z", v: "0.7.27" },
  { timestamp: "2023-03-11T01:45:36.206Z", v: "0.7.28" },
  { timestamp: "2023-03-12T03:38:31.629Z", v: "0.7.29" },
  { timestamp: "2023-03-21T21:25:32.360Z", v: "0.7.30" },
  { timestamp: "2023-03-25T01:20:26.530Z", v: "0.7.31" },
  { timestamp: "2023-03-27T23:16:34.558Z", v: "0.7.32" },
  { timestamp: "2023-03-29T17:26:14.254Z", v: "0.7.33" },
  { timestamp: "2023-03-30T01:36:40.595Z", v: "0.7.34" },
  { timestamp: "2023-04-02T02:52:03.703Z", v: "0.7.35" },
  { timestamp: "2023-04-08T02:46:17.420Z", v: "0.7.36" },
  { timestamp: "2023-04-15T04:57:30.122Z", v: "0.7.37" },
  { timestamp: "2023-04-20T19:16:07.660Z", v: "0.7.38" },
  { timestamp: "2023-05-01T20:11:43.387Z", v: "0.7.39" },
  { timestamp: "2023-05-16T17:57:37.742Z", v: "0.7.40" },
  { timestamp: "2023-06-01T22:17:15.601Z", v: "0.7.41" },
  { timestamp: "2023-06-06T17:09:26.270Z", v: "0.7.42" },
  { timestamp: "2023-06-16T22:49:55.949Z", v: "0.7.43" },
  { timestamp: "2023-06-23T18:45:55.217Z", v: "0.7.44" },
  { timestamp: "2023-07-02T11:01:14.165Z", v: "0.7.45" },
  { timestamp: "2023-07-13T18:44:03.522Z", v: "0.7.46" },
  { timestamp: "2023-08-01T21:42:08.274Z", v: "0.7.47" },
  { timestamp: "2023-08-21T02:17:46.358Z", v: "0.7.48" },
  { timestamp: "2023-09-01T23:23:47.786Z", v: "0.7.49" },
  { timestamp: "2023-09-08T19:44:55.846Z", v: "0.7.50" },
  { timestamp: "2023-09-15T00:27:25.870Z", v: "0.7.51" },
  { timestamp: "2023-09-16T23:28:57.715Z", v: "0.7.52" },
  { timestamp: "2023-10-02T01:39:54.636Z", v: "0.7.53" },
  { timestamp: "2023-10-15T17:49:21.287Z", v: "0.7.54" },
  { timestamp: "2023-11-02T03:04:34.659Z", v: "0.7.55" },
  { timestamp: "2023-11-13T05:02:06.283Z", v: "0.7.56" },
  { timestamp: "2023-11-18T20:05:26.056Z", v: "0.7.57" },
  { timestamp: "2023-11-21T19:47:53.144Z", v: "0.7.58" },
  { timestamp: "2023-12-02T02:34:17.195Z", v: "0.7.59" },
  { timestamp: "2023-12-17T16:36:18.074Z", v: "0.8.0" },
  { timestamp: "2023-12-17T21:43:56.234Z", v: "0.8.1" },
  { timestamp: "2023-12-20T02:42:49.683Z", v: "0.8.2" },
  { timestamp: "2023-12-30T15:16:52.796Z", v: "0.8.3" },
  { timestamp: "2024-01-03T18:43:58.270Z", v: "0.8.4" },
  { timestamp: "2024-01-19T00:52:20.561Z", v: "0.8.5" },
  { timestamp: "2024-01-19T19:35:12.173Z", v: "0.8.6" },
  { timestamp: "2024-02-02T00:32:53.700Z", v: "0.8.7" },
  { timestamp: "2024-02-13T15:29:47.770Z", v: "0.8.8" },
  { timestamp: "2024-03-01T23:29:26.493Z", v: "0.8.9" },
  { timestamp: "2024-03-12T03:30:41.371Z", v: "0.9.0" },
  { timestamp: "2024-04-02T05:11:12.938Z", v: "0.9.1" },
  { timestamp: "2024-04-07T16:21:40.213Z", v: "0.9.2" },
  { timestamp: "2024-04-22T19:30:08.418Z", v: "0.9.3" },
  { timestamp: "2024-05-05T00:39:16.661Z", v: "0.9.4" },
  { timestamp: "2024-06-01T19:41:44.737Z", v: "0.9.5" },
  { timestamp: "2024-06-11T19:35:56.316Z", v: "0.9.6" },
  { timestamp: "2024-06-12T20:01:40.933Z", v: "0.9.7" },
  { timestamp: "2024-07-01T20:32:54.536Z", v: "0.9.8" },
  { timestamp: "2024-07-11T21:06:27.288Z", v: "0.9.9" },
  { timestamp: "2024-07-22T18:50:44.068Z", v: "0.9.10" },
  { timestamp: "2024-08-01T23:01:54.326Z", v: "0.9.11" },
  { timestamp: "2024-08-13T18:47:44.573Z", v: "0.9.12" },
  { timestamp: "2024-09-04T02:55:50.823Z", v: "0.9.13" },
  { timestamp: "2024-09-10T16:31:33.352Z", v: "0.9.14" },
  { timestamp: "2024-09-12T20:22:22.486Z", v: "0.9.15" },
  { timestamp: "2024-09-20T16:20:04.708Z", v: "0.9.16" },
  { timestamp: "2024-09-21T17:52:38.298Z", v: "0.9.17" },
  { timestamp: "2024-09-22T19:03:40.945Z", v: "0.9.18" },
  { timestamp: "2024-10-01T15:53:25.169Z", v: "0.9.19" },
  { timestamp: "2024-10-07T17:11:48.566Z", v: "0.9.20" },
  { timestamp: "2024-10-18T20:06:59.633Z", v: "0.9.21" },
  { timestamp: "2024-11-04T16:56:47.871Z", v: "0.9.22" },
  { timestamp: "2024-11-18T20:23:57.693Z", v: "0.9.23" },
  { timestamp: "2024-12-02T18:03:49.737Z", v: "0.9.24" },
  { timestamp: "2024-12-09T20:05:37.330Z", v: "0.9.25" },
  { timestamp: "2025-01-01T19:18:37.073Z", v: "0.9.26" }
] as const

const T_RELEASES = RELEASES.map((release) => {
  const { timestamp } = release
  return [new Date(timestamp).getTime() / 1000, release] as const
})

export function getNearest(t: number) {
  const nbrs: Release[] = []

  for (let i = 0; i < T_RELEASES.length; i++) {
    const [t1, release] = T_RELEASES[i]
    if (t < t1) {
      nbrs.push(release)
      break
    }
    nbrs[0] = release
  }

  return nbrs
}

export class VersionManager {
  randoms: Map<string, any>
  sim: Map<string, any>

  constructor() {
    this.randoms = new Map()
    this.sim = new Map()
  }

  async set(ver: string) {
    const [, minor] = ver.split(".")

    if (!this.randoms.has(ver)) {
      this.randoms.set(ver, await import(`@pkmn/randoms-${ver}`))
    }
    const { TeamGenerators } = this.randoms.get(ver)

    ver = { "7": "0.7.59", "8": "0.8.9", "9": "0.9.26" }[minor]!

    if (!this.sim.has(ver)) {
      this.sim.set(ver, await import(`@pkmn/sim-${ver}`))
    }
    const { Teams, Battle } = this.sim.get(ver)

    Teams.setGeneratorFactory(TeamGenerators)

    return { Battle } as { Battle: AnyBattleT }
  }
}
