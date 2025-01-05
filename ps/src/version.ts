import { Battle as Battle_0_8 } from "@pkmn/sim-0.8.9"
import { Battle as Battle_0_9 } from "@pkmn/sim-0.9.26"

export type AnyBattle = Battle_0_8 | Battle_0_9

type Release = {
  timestamp: string
  randoms: string
  sim: string
}

const RELEASES: Release[] = [
  { timestamp: "2023-01-03T21:24:28.389Z", randoms: "0.7.13", sim: "0.7.59" },
  { timestamp: "2023-01-07T21:15:04.470Z", randoms: "0.7.14", sim: "0.7.59" },
  { timestamp: "2023-01-09T20:37:04.417Z", randoms: "0.7.15", sim: "0.7.59" },
  { timestamp: "2023-01-20T03:16:20.080Z", randoms: "0.7.16", sim: "0.7.59" },
  { timestamp: "2023-01-21T01:37:38.914Z", randoms: "0.7.17", sim: "0.7.59" },
  { timestamp: "2023-01-22T03:16:23.113Z", randoms: "0.7.18", sim: "0.7.59" },
  { timestamp: "2023-01-29T19:35:07.841Z", randoms: "0.7.19", sim: "0.7.59" },
  { timestamp: "2023-01-29T21:33:54.921Z", randoms: "0.7.20", sim: "0.7.59" },
  { timestamp: "2023-02-02T00:05:24.933Z", randoms: "0.7.21", sim: "0.7.59" },
  { timestamp: "2023-02-13T19:06:20.842Z", randoms: "0.7.22", sim: "0.7.59" },
  { timestamp: "2023-02-24T03:30:22.915Z", randoms: "0.7.23", sim: "0.7.59" },
  { timestamp: "2023-02-27T00:10:46.032Z", randoms: "0.7.24", sim: "0.7.59" },
  { timestamp: "2023-02-28T16:34:19.064Z", randoms: "0.7.25", sim: "0.7.59" },
  { timestamp: "2023-03-01T23:03:58.373Z", randoms: "0.7.26", sim: "0.7.59" },
  { timestamp: "2023-03-09T23:25:35.423Z", randoms: "0.7.27", sim: "0.7.59" },
  { timestamp: "2023-03-11T01:45:36.206Z", randoms: "0.7.28", sim: "0.7.59" },
  { timestamp: "2023-03-12T03:38:31.629Z", randoms: "0.7.29", sim: "0.7.59" },
  { timestamp: "2023-03-21T21:25:32.360Z", randoms: "0.7.30", sim: "0.7.59" },
  { timestamp: "2023-03-25T01:20:26.530Z", randoms: "0.7.31", sim: "0.7.59" },
  { timestamp: "2023-03-27T23:16:34.558Z", randoms: "0.7.32", sim: "0.7.59" },
  { timestamp: "2023-03-29T17:26:14.254Z", randoms: "0.7.33", sim: "0.7.59" },
  { timestamp: "2023-03-30T01:36:40.595Z", randoms: "0.7.34", sim: "0.7.59" },
  { timestamp: "2023-04-02T02:52:03.703Z", randoms: "0.7.35", sim: "0.7.59" },
  { timestamp: "2023-04-08T02:46:17.420Z", randoms: "0.7.36", sim: "0.7.59" },
  { timestamp: "2023-04-15T04:57:30.122Z", randoms: "0.7.37", sim: "0.7.59" },
  { timestamp: "2023-04-20T19:16:07.660Z", randoms: "0.7.38", sim: "0.7.59" },
  { timestamp: "2023-05-01T20:11:43.387Z", randoms: "0.7.39", sim: "0.7.59" },
  { timestamp: "2023-05-16T17:57:37.742Z", randoms: "0.7.40", sim: "0.7.59" },
  { timestamp: "2023-06-01T22:17:15.601Z", randoms: "0.7.41", sim: "0.7.59" },
  { timestamp: "2023-06-06T17:09:26.270Z", randoms: "0.7.42", sim: "0.7.59" },
  { timestamp: "2023-06-16T22:49:55.949Z", randoms: "0.7.43", sim: "0.7.59" },
  { timestamp: "2023-06-23T18:45:55.217Z", randoms: "0.7.44", sim: "0.7.59" },
  { timestamp: "2023-07-02T11:01:14.165Z", randoms: "0.7.45", sim: "0.7.59" },
  { timestamp: "2023-07-13T18:44:03.522Z", randoms: "0.7.46", sim: "0.7.59" },
  { timestamp: "2023-08-01T21:42:08.274Z", randoms: "0.7.47", sim: "0.7.59" },
  { timestamp: "2023-08-21T02:17:46.358Z", randoms: "0.7.48", sim: "0.7.59" },
  { timestamp: "2023-09-01T23:23:47.786Z", randoms: "0.7.49", sim: "0.7.59" },
  { timestamp: "2023-09-08T19:44:55.846Z", randoms: "0.7.50", sim: "0.7.59" },
  { timestamp: "2023-09-15T00:27:25.870Z", randoms: "0.7.51", sim: "0.7.59" },
  { timestamp: "2023-09-16T23:28:57.715Z", randoms: "0.7.52", sim: "0.7.59" },
  { timestamp: "2023-10-02T01:39:54.636Z", randoms: "0.7.53", sim: "0.7.59" },
  { timestamp: "2023-10-15T17:49:21.287Z", randoms: "0.7.54", sim: "0.7.59" },
  { timestamp: "2023-11-02T03:04:34.659Z", randoms: "0.7.55", sim: "0.7.59" },
  { timestamp: "2023-11-13T05:02:06.283Z", randoms: "0.7.56", sim: "0.7.59" },
  { timestamp: "2023-11-18T20:05:26.056Z", randoms: "0.7.57", sim: "0.7.59" },
  { timestamp: "2023-11-21T19:47:53.144Z", randoms: "0.7.58", sim: "0.7.59" },
  { timestamp: "2023-12-02T02:34:17.195Z", randoms: "0.7.59", sim: "0.7.59" },
  { timestamp: "2023-12-17T16:36:18.074Z", randoms: "0.8.0", sim: "0.8.9" },
  { timestamp: "2023-12-17T21:43:56.234Z", randoms: "0.8.1", sim: "0.8.9" },
  { timestamp: "2023-12-20T02:42:49.683Z", randoms: "0.8.2", sim: "0.8.9" },
  { timestamp: "2023-12-30T15:16:52.796Z", randoms: "0.8.3", sim: "0.8.9" },
  { timestamp: "2024-01-03T18:43:58.270Z", randoms: "0.8.4", sim: "0.8.9" },
  { timestamp: "2024-01-19T00:52:20.561Z", randoms: "0.8.5", sim: "0.8.9" },
  { timestamp: "2024-01-19T19:35:12.173Z", randoms: "0.8.6", sim: "0.8.9" },
  { timestamp: "2024-02-02T00:32:53.700Z", randoms: "0.8.7", sim: "0.8.9" },
  { timestamp: "2024-02-13T15:29:47.770Z", randoms: "0.8.8", sim: "0.8.9" },
  { timestamp: "2024-03-01T23:29:26.493Z", randoms: "0.8.9", sim: "0.8.9" },
  { timestamp: "2024-03-12T03:30:41.371Z", randoms: "0.9.0", sim: "0.9.26" },
  { timestamp: "2024-04-02T05:11:12.938Z", randoms: "0.9.1", sim: "0.9.26" },
  { timestamp: "2024-04-07T16:21:40.213Z", randoms: "0.9.2", sim: "0.9.26" },
  { timestamp: "2024-04-22T19:30:08.418Z", randoms: "0.9.3", sim: "0.9.26" },
  { timestamp: "2024-05-05T00:39:16.661Z", randoms: "0.9.4", sim: "0.9.26" },
  { timestamp: "2024-06-01T19:41:44.737Z", randoms: "0.9.5", sim: "0.9.26" },
  { timestamp: "2024-06-11T19:35:56.316Z", randoms: "0.9.6", sim: "0.9.26" },
  { timestamp: "2024-06-12T20:01:40.933Z", randoms: "0.9.7", sim: "0.9.26" },
  { timestamp: "2024-07-01T20:32:54.536Z", randoms: "0.9.8", sim: "0.9.26" },
  { timestamp: "2024-07-11T21:06:27.288Z", randoms: "0.9.9", sim: "0.9.26" },
  { timestamp: "2024-07-22T18:50:44.068Z", randoms: "0.9.10", sim: "0.9.26" },
  { timestamp: "2024-08-01T23:01:54.326Z", randoms: "0.9.11", sim: "0.9.26" },
  { timestamp: "2024-08-13T18:47:44.573Z", randoms: "0.9.12", sim: "0.9.26" },
  { timestamp: "2024-09-04T02:55:50.823Z", randoms: "0.9.13", sim: "0.9.26" },
  { timestamp: "2024-09-10T16:31:33.352Z", randoms: "0.9.14", sim: "0.9.26" },
  { timestamp: "2024-09-12T20:22:22.486Z", randoms: "0.9.15", sim: "0.9.26" },
  { timestamp: "2024-09-20T16:20:04.708Z", randoms: "0.9.16", sim: "0.9.26" },
  { timestamp: "2024-09-21T17:52:38.298Z", randoms: "0.9.17", sim: "0.9.26" },
  { timestamp: "2024-09-22T19:03:40.945Z", randoms: "0.9.18", sim: "0.9.26" },
  { timestamp: "2024-10-01T15:53:25.169Z", randoms: "0.9.19", sim: "0.9.26" },
  { timestamp: "2024-10-07T17:11:48.566Z", randoms: "0.9.20", sim: "0.9.26" },
  { timestamp: "2024-10-18T20:06:59.633Z", randoms: "0.9.21", sim: "0.9.26" },
  { timestamp: "2024-11-04T16:56:47.871Z", randoms: "0.9.22", sim: "0.9.26" },
  { timestamp: "2024-11-18T20:23:57.693Z", randoms: "0.9.23", sim: "0.9.26" },
  { timestamp: "2024-12-02T18:03:49.737Z", randoms: "0.9.24", sim: "0.9.26" },
  { timestamp: "2024-12-09T20:05:37.330Z", randoms: "0.9.25", sim: "0.9.26" },
  { timestamp: "2025-01-01T19:18:37.073Z", randoms: "0.9.26", sim: "0.9.26" }
] as const

export class VersionManager {
  desc: [number, Release][]
  randoms: Map<string, any>
  sim: Map<string, any>

  constructor() {
    this.desc = [...RELEASES].reverse().map((release) => {
      const { timestamp } = release
      return [new Date(timestamp).getTime() / 1000, release]
    })

    this.randoms = new Map()
    this.sim = new Map()
  }

  async setByUnixSeconds(unix: number) {
    for (const [unix1, { randoms, sim }] of this.desc) {
      if (unix1 < unix) {
        if (!this.randoms.has(randoms)) {
          this.randoms.set(randoms, await import(`@pkmn/randoms-${randoms}`))
        }
        if (!this.sim.has(sim)) {
          this.sim.set(sim, await import(`@pkmn/sim-${sim}`))
        }

        const { Teams, Battle } = this.sim.get(sim)
        const { TeamGenerators } = this.randoms.get(randoms)
        Teams.setGeneratorFactory(TeamGenerators)

        return Battle as typeof Battle_0_8 | typeof Battle_0_9
      }
    }

    return false
  }
}
