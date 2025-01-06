console.log(JSON.stringify(`function (logs) {
            return logs.map((log) => {
              if (log[0] === "sideupdate") {
                const [side, rest] = log[1].split("\n")
                const [, type, v] = rest.split("|")
                return ["sideupdate", side, type, type === "request" ? JSON.parse(v) : v]
              }
              if (log[0] === "end") {
                return ["end", JSON.parse(log[1])]
              }
              return log
            })
          }`))