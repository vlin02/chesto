let res = await fetch(new URL("https://play.pokemonshowdown.com/api/getassertion"), {
  method: "POST",
  body: "userid=chest10&challstr=4|a3a3457813b9bfb5becc73ebbd6c678068f92762036d793f6a61614d8343ef53956a0d3a2f329aa68c05a4f4c72e2a178ff54cc93bf3fee64ae1bc0957e27b79add00d7af03a27a2dcde5d14b5c12e64d511edc995c071751caeebbc64a1b9379db0690430990d0d34530b96446bb497823f1d68a72d56cc5c53884fca04a3ab"
})
console.log(await res.text())