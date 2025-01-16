import x from "/Users/vilin/chesto/ps/src/__tmp/set.json"

const v= Object.values(x).map(x => x.level)
console.log(Math.max(...v), Math.min(...v))