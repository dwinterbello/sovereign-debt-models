// Methodology page figures. Every figure is computed in the browser from the pack, by the same engine the
// dashboard runs (levelGap, evaluate), so a figure cannot disagree with the numbers it explains.
// MATLAB-style axes: boxed, ticks inside on all four sides, light grid, default colour order.
const FIG = (() => {
  const AX = "#262626", FONT = "Helvetica, Arial, sans-serif";
  const C = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F"];
  const has = () => typeof Plotly !== "undefined";
  const axis = (extra = {}) => ({
    showline: true, linecolor: AX, linewidth: 1, mirror: "ticks", ticks: "inside", ticklen: 5, tickcolor: AX,
    showgrid: true, gridcolor: "#e3e3e3", zeroline: false, fixedrange: true,
    tickfont: { family: FONT, size: 11, color: AX }, title: { font: { family: FONT, size: 12, color: AX }, standoff: 8 },
    ...extra, title: { font: { family: FONT, size: 12, color: AX }, standoff: 8, ...(extra.title || {}) },
  });
  // Margins here cover the plot frame and axis labels only; draw() adds the legend's own height underneath,
  // so a long legend grows the figure instead of covering the axis.
  const panel = (title, extra = {}) => layout({
    title: { text: title, font: { size: 12.5 }, x: 0.5, xanchor: "center", y: 0.98, yanchor: "top" },
    margin: { l: 56, r: 12, t: 30, b: 50 },
    ...extra });
  const layout = (extra = {}) => ({
    margin: { l: 62, r: 16, t: 16, b: 50 }, font: { family: FONT, size: 11, color: AX },
    paper_bgcolor: "#fff", plot_bgcolor: "#fff", showlegend: true, hovermode: "x unified",
    legend: { font: { family: FONT, size: 10.5, color: AX } },
    hoverlabel: { bgcolor: "#fff", bordercolor: AX, font: { family: FONT, size: 11, color: AX } },
    ...extra,
    xaxis: axis(extra.xaxis || {}), yaxis: axis(extra.yaxis || {}),
  });
  const cfg = { displayModeBar: false, responsive: true };
  // Legend sits under the axis title, inside the container, and the figure is as tall as plot + axis + legend.
  // Wide panels lay entries out in rows (estimated from name length); narrow panels (three to a row) stack them.
  const draw = (id, traces, lay) => {
    const el = document.getElementById(id);
    if (!el || !has()) return;
    const W = el.clientWidth || 600;
    const inRow = !!(el.parentElement && el.parentElement.classList.contains("row"));
    const names = traces.filter((t) => t.showlegend !== false && t.name).map((t) => t.name);
    // Narrow panels (three-up rows, phones) and crowded legends: hide the legend, keep full names on hover.
    const crowded = names.length > 5 || (inRow && names.length > 3) || W < 420;
    if (crowded) lay.showlegend = false;
    const narrow = W < 520;
    let rows = 0;
    if (lay.showlegend !== false && names.length) {
      if (narrow) rows = names.length;
      else { let used = 0; rows = 1;
        for (const n of names) { const w = n.length * 6 + 34; if (used + w > W - 30 && used > 0) { rows++; used = 0; } used += w; } }
    }
    const legH = rows ? rows * 15 + 14 : 0;
    const m = lay.margin || {};
    lay.margin = { ...m, b: Math.max(m.b ?? 48, 48) + legH, t: m.t ?? 30 };
    if (lay.showlegend !== false) {
      lay.legend = { ...(lay.legend || {}), orientation: narrow ? "v" : "h", x: 0, xanchor: "left", y: 0, yref: "container",
        yanchor: "bottom", bordercolor: AX, borderwidth: 0.75, tracegroupgap: 2 };
    }
    const plotH = el.classList.contains("tall") ? 400 : 290;
    lay.height = plotH + (lay.margin.t ?? 16) + lay.margin.b;
    el.style.height = `${lay.height}px`;
    try { Plotly.newPlot(el, traces, lay, cfg); } catch (e) { el.textContent = `Figure failed: ${e.message}`; }
  };
  return { AX, FONT, C, axis, layout, panel, draw };
})();

const sum = (a) => a.reduce((x, y) => x + y, 0);
const fmtbn = (v) => `${v < 0 ? "−" : ""}${Math.abs(v).toFixed(1)}bn`;

function figGap() {
  const tau = Array.from({ length: 48 }, (_, i) => i + 1);
  const lv = tau.map((t) => levelGap(1, "level", t)), gr = tau.map((t) => levelGap(1, "growth", t));
  FIG.draw("fig-gap", [
    { x: tau, y: lv, name: "One-off level shift, s = 1%", mode: "lines", line: { color: FIG.C[0], width: 2 } },
    { x: tau, y: gr, name: "Shock every year, s = 1pp", mode: "lines", line: { color: FIG.C[1], width: 2 } },
  ], FIG.layout({ xaxis: { title: { text: "τ, months the shock has been in force" }, dtick: 12, range: [0, 49] },
                  yaxis: { title: { text: "g, driver above baseline (%)" }, rangemode: "tozero" },
                  annotations: [[lv, FIG.C[0], "bottom"], [gr, FIG.C[1], "top"]].map(([y, col, va]) => ({
                    x: 48, y: y[47], text: `g = ${y[47].toFixed(2)}%`, showarrow: false, xanchor: "right", yanchor: va,
                    font: { size: 10.5, color: col } })) }));
}

function figIdentity() {
  const cc = "fr", c = PACK.countries[cc];
  const pr = PACK.presets.find((p) => p.id === "stagflation") || PACK.presets[1];
  if (!c || !pr) return;
  const shocks = JSON.parse(JSON.stringify(pr.shocks));
  const r = evaluate(PACK, cc, shocks);
  const tr = c.lines.filter((ln) => r.delta_by_line[ln.id] && r.delta_by_line[ln.id].some((v) => Math.abs(v) > 1e-9))
    .map((ln, i) => ({ type: "bar", name: `σΔL: ${ln.label} (σ = ${ln.sign > 0 ? "+1" : "−1"})`, x: c.periods,
      y: r.delta_by_line[ln.id].map((v) => ln.sign * v), marker: { color: FIG.C[i % 7], line: { color: FIG.AX, width: 0.75 } } }));
  if (r.feedback.some((v) => v)) tr.push({ type: "bar", name: "F: interest on extra debt", x: c.periods, y: r.feedback,
    marker: { color: FIG.C[5], line: { color: FIG.AX, width: 0.75 } } });
  tr.push({ type: "scatter", mode: "lines+markers", name: "ΔB: total shown on the dashboard", x: c.periods, y: r.delta,
    line: { color: FIG.AX, width: 1.5 }, marker: { symbol: "diamond", size: 7, color: FIG.AX } });
  document.getElementById("cap-identity").textContent =
    `${c.name}, preset "${pr.label}": each line's change times its sign σ, stacked; the diamonds are ΔB. ` +
    (() => { const g = Math.max(...r.delta.map((d, i) => Math.abs(d - sum(tr.slice(0, -1).map((t) => t.y[i])))));
             return `They add up exactly (largest gap ${g < 1e-12 ? "below 1e-12" : g.toExponential(0)} bn).`; })();
  FIG.draw("fig-identity", tr, FIG.layout({ barmode: "relative", bargap: 0.45,
    xaxis: { title: { text: "p, calendar year" }, type: "category", showgrid: false },
    yaxis: { title: { text: `ΔB, change in borrowing (${c.unit})` }, zeroline: true, zerolinecolor: FIG.AX } }));
}

function figLevel() {
  const cc = "fr", c = PACK.countries[cc];
  if (!c) return;
  const lv = evaluate(PACK, cc, { ngdp: { size: 1, mode: "level" } });
  const gr = evaluate(PACK, cc, { ngdp: { size: 1, mode: "growth" } });
  const rec = (r) => r.delta_by_line[c.lines.find((l) => l.sign < 0).id];
  FIG.draw("fig-level", [
    { type: "bar", name: "GDP level 1% higher, once (g = 1%)", x: c.periods, y: rec(lv), marker: { color: FIG.C[0], line: { color: FIG.AX, width: 0.75 } } },
    { type: "bar", name: "GDP growth 1pp higher every year (g compounds)", x: c.periods, y: rec(gr), marker: { color: FIG.C[1], line: { color: FIG.AX, width: 0.75 } } },
  ], FIG.layout({ barmode: "group", bargap: 0.35, xaxis: { title: { text: "p, calendar year" }, type: "category", showgrid: false },
                  yaxis: { title: { text: `Δy, change in ${c.name} receipts (${c.unit})` } } }));
}

function figForecast() {
  const ccs = Object.keys(PACK.countries).filter((cc) => PACK.countries[cc].history);
  ccs.forEach((cc) => {
    const c = PACK.countries[cc], h = c.history, b = c.baseline, st = h.stub || { total: 0 };
    const full = b.total.map((v, i) => v + (i === 0 ? st.total : 0));
    const x = [...h.periods, ...c.periods];
    const tr = [
      { x: h.periods, y: h.total, name: "Outturn", mode: "lines+markers", line: { color: FIG.AX, width: 1.5 }, marker: { size: 4, color: FIG.AX } },
      { x: [h.periods[h.periods.length - 1], ...c.periods], y: [h.total[h.total.length - 1], ...full], name: "Our forecast",
        mode: "lines+markers", line: { color: FIG.C[0], width: 2 }, marker: { size: 6, color: "#fff", line: { color: FIG.C[0], width: 1.5 } } },
    ];
    if (b.with_policy) tr.push({ x: [h.periods[h.periods.length - 1], ...c.periods],
      y: [h.total[h.total.length - 1], ...b.with_policy.map((v, i) => v + (i === 0 ? st.total : 0))],
      name: "Our forecast + legislated policy", mode: "lines+markers", line: { color: FIG.C[4], width: 2 },
      marker: { size: 6, symbol: "square", color: FIG.C[4] } });
    if (b.board_full_year) tr.push({ x: c.periods, y: b.board_full_year, name: "Our cash board", mode: "lines",
      line: { color: FIG.C[1], width: 1.5, dash: "dash" } });
    const a = (b.authority || {}).years || {};
    const ap = c.periods.filter((p) => a[p] !== undefined);
    if (ap.length) tr.push({ x: ap, y: ap.map((p) => a[p]), name: "Official forecast (not used)", mode: "markers",
      marker: { symbol: "diamond", size: 9, color: FIG.C[3] } });
    FIG.draw(`fig-forecast-${cc}`, tr, FIG.panel(c.name, {
      xaxis: { type: "category", range: [-0.5, x.length - 0.5], nticks: 8 },
      yaxis: { title: { text: `B, borrowing measure (${c.unit})` } },
      shapes: [{ type: "rect", xref: "x", yref: "paper", x0: h.periods.length - 0.5, x1: x.length - 0.5, y0: 0, y1: 1,
                 fillcolor: "#f2f2f2", line: { width: 0 }, layer: "below" }] }));
  });
}

function figShares() {
  ["fr", "de", "it"].forEach((cc) => {
    const c = PACK.countries[cc];
    const sh = c && c.model && c.model.shares;
    if (!sh) return;
    const names = { share: "persists", share_xc: "persists, Covid out", hold: "held at last year's", revert: "reverts" };
    const tr = Object.entries(sh).map(([ln, s], i) => {
      const yrs = Object.keys(s.share).sort();
      const base = yrs.filter((y) => y < s.first_forecast_year).pop();
      const v0 = s.share[base];
      return { x: yrs, y: yrs.map((y) => (100 * s.share[y]) / v0), mode: "lines+markers",
        name: `${s.label} ÷ ${s.driver === "ngdp" ? "GDP" : "HICP"}: ${names[s.method] || s.method}`,
        line: { color: FIG.C[i % 7], width: 1.75 }, marker: { size: 5, color: FIG.C[i % 7] } };
    });
    const first = Object.values(sh)[0].first_forecast_year;
    const xs = Object.keys(Object.values(sh)[0].share).sort();
    const i0 = xs.indexOf(first);
    FIG.draw(`fig-shares-${cc}`, tr, FIG.panel(c.name, {
      margin: { l: 56, r: 12, t: 30, b: 92 },
      xaxis: { type: "category", tickangle: -90 }, yaxis: { title: { text: "ω, share of driver (last outturn year = 100)" } },
      shapes: i0 >= 0 ? [{ type: "rect", xref: "x", yref: "paper", x0: i0 - 0.5, x1: xs.length - 0.5, y0: 0, y1: 1,
                           fillcolor: "#f2f2f2", line: { width: 0 }, layer: "below" }] : [] }));
  });
}

function figBacktest() {
  const my = (PACK.model_selection || {}).multi_year;
  if (!my) return;
  const specs = [["final", "In use", FIG.C[0]], ["current", "Before Nov 2026", FIG.C[1]], ["naive", "Last year's share of GDP", FIG.C[2]]];
  ["fr", "de", "it"].forEach((cc) => {
    if (!my[cc]) return;
    const tr = specs.filter(([s]) => my[cc][s]).map(([s, label, col]) => ({ type: "bar", name: label,
      x: ["1 year", "2 years", "3 years"], y: [1, 2, 3].map((k) => my[cc][s][`h${k}`].mae_total_ex_covid),
      marker: { color: col, line: { color: FIG.AX, width: 0.75 } } }));
    FIG.draw(`fig-bt-${cc}`, tr, FIG.panel(PACK.countries[cc].name, { barmode: "group", bargap: 0.3,
      xaxis: { type: "category", showgrid: false, title: { text: "Horizon (years ahead)" } }, yaxis: { title: { text: "MAE of B ex interest (EUR bn)" }, rangemode: "tozero" } }));
  });
  const rows = (PACK.model_selection || {}).multi_year_rows || {};
  ["fr", "de", "it"].forEach((cc) => {
    const r = (rows[cc] || []).filter((x) => x[2] === 2);
    if (!r.length) return;
    const tr = [["final", "In use", FIG.C[0]], ["current", "Before Nov 2026", FIG.C[1]]].map(([s, label, col]) => {
      const pts = r.filter((x) => x[3] === s).sort((a, b) => a[1] - b[1]);
      return { x: pts.map((x) => x[1]), y: pts.map((x) => x[4]), name: label, mode: "lines+markers",
        line: { color: col, width: 1.75 }, marker: { size: 6, color: col } };
    });
    FIG.draw(`fig-err-${cc}`, tr, FIG.panel(PACK.countries[cc].name, {
      xaxis: { title: { text: "Target year (forecast made two years earlier)" }, dtick: 2 },
      yaxis: { title: { text: "Forecast − outturn (EUR bn)" }, zeroline: true, zerolinecolor: FIG.AX },
      shapes: [{ type: "rect", xref: "x", yref: "paper", x0: 2019.5, x1: 2021.5, y0: 0, y1: 1, fillcolor: "#e6e6e6",
                 line: { width: 0 }, layer: "below" }] }));
  });
}

function figElasticity() {
  const g = Array.from({ length: 41 }, (_, i) => -10 + i * 0.5);
  const tr = [0.5, 1, 1.5].map((e, i) => ({ x: g, y: g.map((v) => 100 * (Math.pow(1 + v / 100, e) - 1)), mode: "lines",
    name: `ε = ${e}`, line: { color: FIG.C[i], width: 2 } }));
  const used = [];
  for (const cc of ["es", "se", "no", "jp"]) {
    const c = PACK.countries[cc];
    if (!c) continue;
    for (const ch of c.channels) if (ch.type && ch.type.startsWith("elasticity") && ch.elasticity !== undefined)
      used.push([`${c.name}: ${ch.line} on ${ch.driver}`, ch.elasticity]);
  }
  tr.push({ x: used.map(() => 5), y: used.map(([, e]) => 100 * (Math.pow(1.05, e) - 1)), mode: "markers", name: "In use, at a 5% shock",
    text: used.map(([n, e]) => `${n}: ε = ${(+e).toFixed(2)}`), hovertemplate: "%{text}<extra></extra>",
    marker: { symbol: "diamond", size: 8, color: FIG.AX } });
  FIG.draw("fig-elasticity", tr, FIG.layout({ hovermode: "closest", xaxis: { title: { text: "Driver gap g, %" } },
    yaxis: { title: { text: "Line response (1+g/100)^ε − 1, %" }, zeroline: true, zerolinecolor: FIG.AX } }));
}

function heat(id, ch, title) {
  const n = ch.coef.length, yrs = Array.from({ length: n }, (_, i) => `Year ${i + 1}`);
  FIG.draw(id, [{ type: "heatmap", z: ch.coef, x: yrs, y: yrs, colorscale: [[0, "#f7fbff"], [1, "#0072BD"]],
    text: ch.coef.map((r) => r.map((v) => v.toFixed(2))), texttemplate: "%{text}", hovertemplate: "shock %{y}, effect %{x}: %{z:.3f}<extra></extra>",
    colorbar: { thickness: 10, outlinecolor: FIG.AX, outlinewidth: 0.75 } }],
  FIG.layout({ title: { text: title, font: { size: 12 }, x: 0.5, y: 0.97 }, showlegend: false, hovermode: "closest", margin: { l: 66, r: 16, t: 40, b: 48 },
    xaxis: { title: { text: "p, year the effect shows up" }, showgrid: false }, yaxis: { title: { text: "j, year of the shock" }, autorange: "reversed", showgrid: false } }));
}

function figMatrix() {
  const uk = PACK.countries.uk;
  if (!uk) return;
  const a = uk.channels.find((ch) => ch.type === "matrix" && ch.driver === "ngdp" && ch.line === "receipts");
  const b = uk.channels.find((ch) => ch.type === "matrix" && ch.kind === "persistent" && ch.line === "interest");
  if (a) heat("fig-matrix-a", a, "Receipts per 1% nominal GDP level, £bn");
  if (b) heat("fig-matrix-b", b, "Debt interest per 1bp, persistent, £bn");
}

function figLinear() {
  const DN = { ngdp: "GDP", cpi: "CPI", rates: "yields" };
  const groups = [];
  for (const [cc, c] of Object.entries(PACK.countries)) {
    for (const ch of c.channels) {
      if (ch.type !== "linear" || !ch.per_unit) continue;
      const v = ch.per_unit.growth || ch.per_unit.level;
      if (!v || !v.some((x) => Math.abs(x) > 1e-12)) continue;
      groups.push({ cc, c, ch, v });
    }
  }
  const tr = groups.map(({ c, ch, v }, i) => {
    const per = ch.driver === "rates" ? 100 : 1;
    return { type: "bar", x: c.periods.map((_, k) => `Year ${k + 1}`), y: v.map((x) => x * per),
      name: `${c.name === "United States" ? "US" : c.name} ${ch.badge.split(" ")[0]}: ${DN[ch.driver] || ch.driver} → ${ch.line}${per === 100 ? " (per 100bp)" : " (per 1pp)"}`,
      marker: { color: FIG.C[i % 7], line: { color: FIG.AX, width: 0.75 } }, opacity: i >= 7 ? 0.6 : 1 };
  });
  FIG.draw("fig-linear", tr, FIG.layout({ barmode: "group", bargap: 0.2,
    xaxis: { type: "category", showgrid: false, title: { text: "Years after the shock starts" } },
    yaxis: { title: { text: "R, response per unit of shock (local currency bn)" }, zeroline: true, zerolinecolor: FIG.AX } }));
}

function figBook() {
  ["fr", "de", "it"].forEach((cc) => {
    const c = PACK.countries[cc];
    const g = c && c.channels.find((ch) => ch.type === "grid" && ch.driver === "rates");
    if (!g) return;
    const tr = c.periods.map((p, k) => ({ x: g.knots, y: g.values.map((row) => row[k]), mode: "lines+markers", name: p,
      line: { color: FIG.C[k % 7], width: 1.75 }, marker: { size: 6, color: FIG.C[k % 7] } }));
    FIG.draw(`fig-book-${cc}`, tr, FIG.panel(c.name, {
      hovermode: "closest", xaxis: { title: { text: "Δr, parallel yield shift (bp)" }, zeroline: true, zerolinecolor: FIG.AX, dtick: 100 },
      yaxis: { title: { text: `ΔI, change in interest (${c.unit})` }, zeroline: true, zerolinecolor: FIG.AX } }));
  });
}

function figCurve() {
  const cf = PACK.curve_factors;
  if (!cf) return;
  const mk = Object.entries(cf.markets).filter(([, f]) => !f.error);
  const one = (key, id, title) => FIG.draw(id, mk.map(([m, f], i) => ({ x: f.grid, y: f[key].map((w) => 100 * w), mode: "lines+markers",
    name: m.toUpperCase(), line: { color: FIG.C[i % 7], width: 1.75 }, marker: { size: 5, color: FIG.C[i % 7] } })),
  FIG.panel(title, { hovermode: "closest",
    xaxis: { title: { text: "t, maturity (years, log scale)" }, type: "log", tickvals: [0.25, 1, 2, 5, 10, 30], ticktext: ["¼", "1", "2", "5", "10", "30"] },
    yaxis: { title: { text: "s(t), yield move (bp)" }, zeroline: true, zerolinecolor: FIG.AX } }));
  one("w3m", "fig-curve-3m", "3-month +100bp, 10-year unchanged");
  one("w10y", "fig-curve-10y", "10-year +100bp, 3-month unchanged");
}

function figFeedback() {
  const cc = PACK.countries.uk ? "uk" : Object.keys(PACK.countries)[0], c = PACK.countries[cc];
  if (!c.feedback) return;
  const s = { rates: { size: 100 }, ngdp: { size: -1, mode: "level" } };
  const off = evaluate(PACK, cc, { ...s, feedback: false }), on = evaluate(PACK, cc, { ...s, feedback: true });
  FIG.draw("fig-feedback", [
    { type: "bar", name: "ΔB, shock alone (F off)", x: c.periods, y: off.delta, marker: { color: FIG.C[0], line: { color: FIG.AX, width: 0.75 } } },
    { type: "bar", name: "ΔB, with F (interest on extra debt)", x: c.periods, y: on.delta, marker: { color: FIG.C[1], line: { color: FIG.AX, width: 0.75 } } },
  ], FIG.layout({ barmode: "group", bargap: 0.35, xaxis: { type: "category", showgrid: false,
    title: { text: c.period_kind.charAt(0).toUpperCase() + c.period_kind.slice(1) } },
    yaxis: { title: { text: `ΔB, change in borrowing (${c.unit})` }, zeroline: true, zerolinecolor: FIG.AX } }));
  document.getElementById("cap-feedback").textContent =
    `${c.name}: yields +100bp and nominal GDP 1% lower. The gap between the bars is F; it compounds because each year's extra borrowing is borrowed again.`;
}

function figChecks() {
  const ck = PACK.checks || {};
  const rows = [];
  for (const [cc, r] of Object.entries(ck)) {
    if (cc.startsWith("_")) continue;
    const errs = [];
    (r.vs_full_model_bn || []).forEach((x) => errs.push(x.max_abs_err));
    Object.values(r.vs_published_table_bn || {}).forEach((x) => errs.push(x));
    if (cc === "uk") Object.values(r).forEach((x) => x && x.max_abs_err_bn !== undefined && errs.push(x.max_abs_err_bn));
    if (errs.length) rows.push([PACK.countries[cc] ? PACK.countries[cc].name : cc, Math.max(...errs.map((e) => Math.max(e, 1e-16)))]);
  }
  FIG.draw("fig-checks", [{ type: "bar", x: rows.map((r) => r[0]), y: rows.map((r) => r[1]), name: "Largest gap, dashboard vs full model",
    marker: { color: FIG.C[0], line: { color: FIG.AX, width: 0.75 } }, hovertemplate: "%{x}: %{y:.1e} bn<extra></extra>" }],
  FIG.layout({ showlegend: false, xaxis: { type: "category", showgrid: false }, yaxis: { type: "log", title: { text: "Largest gap, bn (log scale)" }, exponentformat: "power" },
    shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 1e-9, y1: 1e-9, line: { color: FIG.C[6], width: 1, dash: "dash" } }],
    annotations: [{ xref: "paper", x: 1, y: Math.log10(1e-9), yref: "y", text: "test tolerance 1e-9", showarrow: false, xanchor: "right", yanchor: "bottom", font: { size: 10.5 } }] }));
}

function figBridge() {
  const br = PACK.authority_bridge || {};
  const lab = { receipts: "Receipts", primary: "Spending ex interest", interest: "Interest" };
  for (const [cc, b] of Object.entries(br)) {
    if (b.total_only) continue;
    const c = PACK.countries[cc], x = b.rows.map((r) => r.period);
    const tr = ["receipts", "primary", "interest"].map((k, i) => ({ type: "bar", name: lab[k], x,
      y: b.rows.map((r) => r.lines[k].forecast), marker: { color: FIG.C[i], line: { color: "#fff", width: 1 } },
      hovertemplate: `${lab[k]}  %{y:.1f}bn<extra></extra>` }));
    tr.push({ type: "bar", name: "Definitions", x, y: b.rows.map((r) => ["receipts", "primary", "interest"].reduce((a, k) => a + r.lines[k].definitions, 0)),
      marker: { color: "#b0b0b0", line: { color: "#fff", width: 1 } }, hovertemplate: "Definitions  %{y:.1f}bn<extra></extra>" });
    tr.push({ type: "scatter", mode: "lines+markers", name: "Total gap (ours − theirs)", x, y: b.rows.map((r) => r.gap),
      line: { color: FIG.AX, width: 1.5 }, marker: { symbol: "diamond", size: 7, color: FIG.AX },
      hovertemplate: "<b>Gap  %{y:.1f}bn</b><extra></extra>" });
    FIG.draw(`fig-bridge-${cc}`, tr, FIG.panel(`${c.name} vs ${b.label}`, { barmode: "relative", bargap: 0.4,
      xaxis: { type: "category", showgrid: false }, yaxis: { title: { text: `Ours − authority (${c.unit}; + ours higher)` },
        zeroline: true, zerolinecolor: FIG.AX } }));
  }
}

function figPolicy() {
  for (const [cc, c] of Object.entries(PACK.countries)) {
    const p = c.policy;
    if (!p) continue;
    const grey = { other_tax: "#b0b0b0", other_spend: "#6e6e6e", other_indirect: "#d4d4d4" };
    let k = 0;
    // Legend entries must fit a one-third-width panel: drop parentheticals, cut at the first comma/colon if still long;
    // the full label stays in the hover text.
    const short = (s, n = 34) => { let t = s.replace(/\s*\([^)]*\)/g, ""); if (t.length > n) t = t.split(/[,:]/)[0];
      return t.length > n ? `${t.slice(0, n - 1)}…` : t; };
    const tr = p.groups.map((g) => ({ type: "bar", name: g.netting ? "Netted off (already in base year)" : short(g.label), x: c.periods, y: g.values,
      marker: g.netting ? { color: "#fff", line: { color: FIG.AX, width: 1 }, pattern: { shape: "/", fgcolor: "#6e6e6e", size: 6, solidity: 0.25 } }
        : { color: g.other ? grey[g.id] : FIG.C[k++ % 6], line: { color: "#fff", width: 1 } },
      hovertemplate: `${g.label}  %{y:.1f}bn<extra></extra>` }));
    tr.push({ type: "scatter", mode: "lines+markers", name: "Net of all measures", x: c.periods, y: p.total,
      line: { color: FIG.AX, width: 1.5 }, marker: { symbol: "diamond", size: 7, color: FIG.AX },
      hovertemplate: "<b>Net  %{y:.1f}bn</b><extra></extra>" });
    FIG.draw(`fig-policy-${cc}`, tr, FIG.panel(c.name, { barmode: "relative", bargap: 0.4,
      xaxis: { type: "category", showgrid: false }, yaxis: { title: { text: `c, effect on borrowing (${c.unit}; + adds)` },
        zeroline: true, zerolinecolor: FIG.AX } }));
  }
}

function figOwn() {
  const oa = (PACK.model_selection || {}).own_annual;
  if (!oa) return;
  const ccs = Object.keys(oa).filter((cc) => oa[cc].lines);
  for (const line of ["receipts", "primary"]) {
    const ms = line === "primary" ? ["persist", "revert", "persist_gdp"] : ["persist", "revert"];
    const nm = { persist: "Persist", revert: "Revert to mean", persist_gdp: "Persist, share of GDP" };
    const tr = ms.map((m, i) => ({ type: "bar", name: nm[m],
      x: ccs.map((cc) => PACK.countries[cc] ? PACK.countries[cc].name : cc),
      y: ccs.map((cc) => (oa[cc].lines[line].scores[m] || {}).mean_pct_gdp),
      marker: { color: ccs.map((cc) => (oa[cc].lines[line].chosen === m ? FIG.C[i] : "#ffffff")),
                line: { color: FIG.C[i], width: 1.5 } } }));
    FIG.draw(`fig-own-${line}`, tr, FIG.panel(line === "receipts" ? "Receipts r·Y: share of nominal GDP (solid = rule in use)" : "Primary spending q·P: share of prices (solid = rule in use)",
      { barmode: "group", bargap: 0.3, margin: { l: 56, r: 12, t: 30, b: 90 },
        xaxis: { type: "category", showgrid: false, tickangle: -30 }, yaxis: { title: { text: "Mean absolute error (% of GDP)" }, rangemode: "tozero" } }));
  }
}

function drawMethodologyFigures() {
  for (const f of [figBridge, figPolicy, figOwn, figGap, figIdentity, figLevel, figForecast, figShares, figBacktest, figElasticity, figMatrix, figLinear, figBook, figCurve, figFeedback, figChecks]) {
    try { f(); } catch (e) { console.error(f.name, e); }
  }
}
