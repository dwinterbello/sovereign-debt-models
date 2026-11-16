// Test helper: read {pack, cases:[{cc, shocks}]} on stdin, print the engine's results as JSON.
import { evaluate } from "../../web/engine.js";
let raw = "";
process.stdin.on("data", (c) => (raw += c));
process.stdin.on("end", () => {
  const { pack, cases } = JSON.parse(raw);
  const out = cases.map(({ cc, shocks }) => {
    try { return { ok: true, result: evaluate(pack, cc, shocks) }; }
    catch (e) { return { ok: false, error: String(e.message) }; }
  });
  process.stdout.write(JSON.stringify(out));
});
