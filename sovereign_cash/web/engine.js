// Scenario engine: a line-for-line port of sovereign_cash/scenarios/engine.py.
// Parity with the Python is tested (tests/test_engine_parity.py); change both together.
// Positive delta = more borrowing. No DOM code here: the page imports evaluate().

export const SCHEMA_VERSION = 1;
export const MODES = ["level", "growth"];

export class PackError extends Error {}

export function levelGap(size, mode, tau) {
  if (tau <= 0 || size === 0) return 0;
  if (mode === "level") return size;
  if (mode === "growth") return 100 * (Math.pow(1 + size / 100, tau / 12) - 1);
  throw new PackError(`unknown mode ${mode}`);
}

function power(gap, eps) {
  if (gap === 0) return 0;
  if (gap <= -100) throw new PackError("a level gap of -100% or less has no constant-elasticity response");
  return Math.pow(1 + gap / 100, eps) - 1;
}

function interpGrid(knots, values, x) {
  if (knots.length !== values.length || knots.length < 2) throw new PackError("grid needs at least two knots, one value row per knot");
  for (let j = 1; j < knots.length; j++) if (!(knots[j] > knots[j - 1])) throw new PackError("grid knots must be strictly increasing");
  let i = 0;
  while (i < knots.length - 2 && x > knots[i + 1]) i += 1;
  const x0 = knots[i], x1 = knots[i + 1];
  const w = (x - x0) / (x1 - x0);
  return values[i].map((a, k) => a + w * (values[i + 1][k] - a));
}

function readShock(shocks, driver) {
  const s = shocks[driver] || {};
  const size = Number(s.size || 0);
  const mode = s.mode || "level";
  if (!MODES.includes(mode)) throw new PackError(`shock ${driver}: unknown mode ${mode}`);
  if (!Number.isFinite(size)) throw new PackError(`shock ${driver}: size must be finite`);
  return [size, mode];
}

export function channelDelta(ch, shocks, nPeriods, monthPeriod) {
  const [size, mode] = readShock(shocks, ch.driver);
  const out = new Array(nPeriods).fill(0);
  if (size === 0) return out;
  if (ch.type === "beta_monthly") {
    if (!monthPeriod) throw new PackError(`channel ${ch.id}: beta_monthly needs month_period`);
    const beta = Number(ch.beta);
    for (let m = 0; m < Math.min(ch.tau.length, monthPeriod.length); m++) {
      const p = monthPeriod[m];
      if (p === null || p < 0) continue;
      out[p] += beta * levelGap(size, mode, ch.tau[m]);
    }
    return out;
  }
  if (ch.type === "linear") {
    const perUnit = ch.per_unit[mode];
    if (!perUnit) throw new PackError(`channel ${ch.id}: no '${mode}' response`);
    return perUnit.map((v) => size * Number(v));
  }
  if (ch.type === "grid") return interpGrid(ch.knots.map(Number), ch.values.map((r) => r.map(Number)), size);
  if (ch.type === "elasticity") {
    const eps = Number(ch.elasticity);
    return ch.base.map((b, i) => {
      const taus = ch.tau_months[i];
      let s = 0;
      for (const t of taus) s += power(levelGap(size, mode, t), eps);
      return (Number(b) * s) / taus.length;
    });
  }
  if (ch.type === "elasticity_monthly") {
    if (!monthPeriod) throw new PackError(`channel ${ch.id}: elasticity_monthly needs month_period`);
    const eps = Number(ch.elasticity);
    for (let m = 0; m < Math.min(ch.base.length, monthPeriod.length); m++) {
      const p = monthPeriod[m];
      if (p === null || p < 0) continue;
      out[p] += Number(ch.base[m]) * power(levelGap(size, mode, ch.tau[m]), eps);
    }
    return out;
  }
  if (ch.type === "matrix") {
    const x = [];
    let first = true;
    ch.tau.forEach((tau, j) => {
      const w = Number((ch.exposure || ch.tau.map(() => 1))[j]);
      let v;
      if (ch.kind === "level") v = levelGap(size, mode, tau);
      else if (ch.kind === "persistent") v = tau > 0 ? size : 0;
      else if (ch.kind === "rate") {
        if (tau <= 0) v = 0;
        else if (mode === "growth" || first) { v = size; first = false; }
        else v = 0;
      } else throw new PackError(`channel ${ch.id}: matrix kind must be level, rate or persistent`);
      x.push(w * v);
    });
    const out = new Array(nPeriods).fill(0);
    for (let p = 0; p < nPeriods; p++) for (let j = 0; j < x.length; j++) out[p] += x[j] * Number(ch.coef[j][p]);
    return out;
  }
  throw new PackError(`channel ${ch.id}: unknown type ${ch.type}`);
}

function feedback(fb, pre, ratesBp) {
  const out = [];
  let stock = 0;
  pre.forEach((d, i) => {
    const k = (Number(fb.rate_pct[i]) + ratesBp / 100) / 100;
    const f = (k * (stock + 0.5 * d)) / (1 - 0.5 * k);
    out.push(f);
    stock += d + f;
  });
  return out;
}

export function evaluate(pack, cc, shocks = {}) {
  if (pack.schema_version !== SCHEMA_VERSION) throw new PackError(`pack schema ${pack.schema_version} != engine ${SCHEMA_VERSION}`);
  const c = pack.countries[cc];
  const periods = c.periods;
  const n = periods.length;
  const signs = Object.fromEntries(c.lines.map((ln) => [ln.id, Number(ln.sign)]));
  const byLine = Object.fromEntries(Object.keys(signs).map((k) => [k, new Array(n).fill(0)]));
  const byDriver = {};
  const methods = { ...(c.default_methods || {}), ...(shocks.methods || {}) };
  for (const ch of c.channels) {
    const group = ch.method_group ?? null;
    if (group !== null) {
      const chosen = methods[ch.driver];
      if (chosen === undefined || chosen === null) throw new PackError(`${cc}: driver ${ch.driver} has several methods and no default`);
      if (group !== chosen) continue;
    }
    let d = channelDelta(ch, shocks, n, c.month_period || null);
    const k = Number((shocks.scale || {})[ch.id] ?? 1);
    if (!(k >= 0 && k <= 5)) throw new PackError(`scale for ${ch.id} must be within [0, 5]`);
    if (k !== 1) d = d.map((v) => k * v);
    if (!(ch.line in byLine)) throw new PackError(`channel ${ch.id} targets unknown line ${ch.line}`);
    const sign = signs[ch.line];
    if (!(ch.driver in byDriver)) byDriver[ch.driver] = new Array(n).fill(0);
    d.forEach((v, i) => {
      byLine[ch.line][i] += v;
      byDriver[ch.driver][i] += sign * v;
    });
  }
  const keys = Object.keys(byLine);
  const pre = [];
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (const k of keys) s += signs[k] * byLine[k][i];
    pre.push(s);
  }
  let fb = new Array(n).fill(0);
  if (shocks.feedback && c.feedback) {
    let ratesBp = readShock(shocks, "rates")[0];
    if (ratesBp === 0) ratesBp = 0.5 * (readShock(shocks, "r3m")[0] + readShock(shocks, "r10y")[0]);
    fb = feedback(c.feedback, pre, ratesBp);
  }
  const total = pre.map((a, i) => a + fb[i]);
  const base = c.baseline.total.map(Number);
  return {
    cc, periods, baseline: base,
    delta_by_line: byLine, delta_by_driver: byDriver,
    feedback: fb, delta: total,
    scenario: base.map((b, i) => b + total[i]),
  };
}
