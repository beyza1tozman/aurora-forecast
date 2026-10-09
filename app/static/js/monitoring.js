// /monitoring: forecast log vs observed Kp. Charts are small inline SVGs.

const $ = (id) => document.getElementById(id);
const HORIZON_COLORS = { 1: "#c4eed6", 3: "#b0a3c4", 6: "#7a8cc4" }; // fades from aurora mint into the sky with lead time
const NS = "http://www.w3.org/2000/svg";

const dt = new Intl.DateTimeFormat(undefined, { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
const dayFmt = new Intl.DateTimeFormat(undefined, { day: "numeric", month: "short" });
const num = (v, d = 3) => (v === null || v === undefined ? "–" : v.toFixed(d));
const signed = (v) => (v === null || v === undefined ? "–" : `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}`);

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

// ---------- tiny SVG chart kit ----------

function svg(tag, attrs = {}, parent) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (parent) parent.appendChild(el);
  return el;
}

function frame(container, { height = 180, yMax, yTicks, yFmt = (v) => v, x0, x1, xTicks }) {
  container.innerHTML = "";
  const W = Math.max(container.clientWidth, 280), H = height;
  const m = { l: 36, r: 12, t: 10, b: 22 };
  const root = svg("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img" }, container);
  const x = (t) => m.l + ((t - x0) / (x1 - x0)) * (W - m.l - m.r);
  const y = (v) => H - m.b - (v / yMax) * (H - m.t - m.b);
  for (const v of yTicks) {
    svg("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: "grid" }, root);
    svg("text", { x: m.l - 6, y: y(v) + 3, class: "axis", "text-anchor": "end" }, root).textContent = yFmt(v);
  }
  for (const t of xTicks) {
    svg("text", { x: x(t), y: H - 6, class: "axis", "text-anchor": "middle" }, root).textContent = dayFmt.format(t);
  }
  return { root, x, y, W, H, m };
}

function dayTicks(x0, x1) {
  const ticks = [];
  const d = new Date(x0);
  d.setHours(0, 0, 0, 0);
  for (d.setDate(d.getDate() + 1); d < x1; d.setDate(d.getDate() + 1)) ticks.push(d.getTime());
  return ticks.length > 8 ? ticks.filter((_, i) => i % 2 === 0) : ticks;
}

function empty(container, text) {
  container.innerHTML = `<p class="chart-empty">${esc(text)}</p>`;
}

// ---------- charts ----------

function observedChart(el, series, x0, x1) {
  if (!series.length) return empty(el, "No observed Kp stored yet. The hourly job adds the last 3 days on its first run.");
  const f = frame(el, { yMax: 9, yTicks: [0, 3, 5, 7, 9], x0, x1, xTicks: dayTicks(x0, x1) });
  const step = 3 * 3600e3;
  const bw = Math.max(1, f.x(x0 + step) - f.x(x0) - 2); // 2px surface gap between bars
  for (const o of series) {
    const t = Date.parse(o.interval_start);
    const level = Math.round(o.kp);
    const cls = level >= 7 ? "bar g3" : level >= 5 ? "bar g1" : "bar";
    const r = svg("rect", { x: f.x(t) + 1, y: f.y(o.kp), width: bw, height: Math.max(f.y(0) - f.y(o.kp), 1), class: cls, rx: Math.min(2, bw / 2) }, f.root);
    svg("title", {}, r).textContent = `${dt.format(t)}: Kp ${o.kp.toFixed(2)}`;
  }
}

function forecastChart(el, series, x0, x1) {
  if (!series.length) return empty(el, "No verified 3-hour forecasts in the last 7 days yet.");
  const top = Math.max(0.2, ...series.map((s) => s.p));
  const yMax = [0.2, 0.5, 1].find((v) => top <= v);
  const ticks = yMax === 1 ? [0, 0.5, 1] : [0, yMax / 2, yMax];
  const f = frame(el, { yMax, yTicks: ticks, yFmt: (v) => `${Math.round(v * 100)}%`, x0, x1, xTicks: dayTicks(x0, x1) });
  const pts = series.map((s) => [f.x(Date.parse(s.interval_start) + 1.5 * 3600e3), f.y(s.p)]);
  svg("path", { d: `M${pts.map((p) => p.join(",")).join("L")}`, class: "line", stroke: HORIZON_COLORS[3] }, f.root);
  series.forEach((s, i) => {
    const hit = Math.round(s.kp_observed) >= 4;
    const t = Date.parse(s.interval_start);
    const label = `${dt.format(t)}: P(Kp ≥ 4) ${Math.round(s.p * 100)}%, observed Kp ${s.kp_observed.toFixed(2)}`;
    if (hit) {
      const c = svg("circle", { cx: pts[i][0], cy: f.y(yMax) + 4, r: 4, class: "event" }, f.root);
      svg("title", {}, c).textContent = label;
    }
    const hitArea = svg("circle", { cx: pts[i][0], cy: pts[i][1], r: 6, class: "hit" }, f.root);
    svg("title", {}, hitArea).textContent = label;
  });
}

function brierChart(el, rows) {
  if (!rows.length) return empty(el, "No verified forecasts yet.");
  const days = [...new Set(rows.map((r) => r.day))].sort();
  const x0 = Date.parse(days[0]), x1 = Math.max(Date.parse(days.at(-1)), x0 + 86400e3);
  const top = Math.max(...rows.map((r) => r.brier), 0.05);
  const yMax = [0.05, 0.1, 0.2, 0.5, 1].find((v) => top <= v);
  const f = frame(el, { yMax, yTicks: [0, yMax / 2, yMax], yFmt: (v) => v.toFixed(2), x0, x1, xTicks: days.length > 1 ? days.map(Date.parse).filter((_, i, a) => a.length <= 8 || i % Math.ceil(a.length / 8) === 0) : [x0] });
  const legend = [];
  for (const h of [1, 3, 6]) {
    const r = rows.filter((row) => row.horizon_h === h);
    if (!r.length) continue;
    const pts = r.map((row) => [f.x(Date.parse(row.day)), f.y(row.brier)]);
    svg("path", { d: `M${pts.map((p) => p.join(",")).join("L")}`, class: "line", stroke: HORIZON_COLORS[h] }, f.root);
    r.forEach((row, i) => {
      const c = svg("circle", { cx: pts[i][0], cy: pts[i][1], r: 4, fill: HORIZON_COLORS[h], class: "pt" }, f.root);
      svg("title", {}, c).textContent = `${row.day}, ${h} h ahead: Brier ${row.brier.toFixed(3)} (${row.n} forecasts)`;
    });
    legend.push(`<span><i class="key" style="background:${HORIZON_COLORS[h]}"></i>${h} h ahead</span>`);
  }
  $("brier-legend").innerHTML = legend.join("");
}

// ---------- page ----------

function renderFacts(d) {
  const c = d.counts;
  const since = c.first_issued ? dt.format(new Date(c.first_issued)) : "–";
  $("facts").innerHTML = `
    <div><dt>Forecasts logged</dt><dd class="num">${c.forecasts_logged}</dd></div>
    <div><dt>Verified</dt><dd class="num">${c.verified}</dd></div>
    <div><dt>Observed intervals</dt><dd class="num">${c.observations}</dd></div>
    <div><dt>Logging since</dt><dd class="num">${since}</dd></div>`;
  $("verified-tag").textContent = `${c.verified} verified forecasts`;
  const note = $("db-note");
  note.hidden = d.database === "postgres";
  note.textContent = "This server stores the log in local SQLite, which is wiped when it restarts. In production the log lives in an external Postgres database.";
}

function renderScores(rows) {
  if (!rows.length) {
    $("scores").innerHTML = `<p class="chart-empty">Nothing to score yet. A forecast can be checked once the Kp for its 3-hour interval has been observed, usually 3–9 hours after it was issued.</p>`;
    return;
  }
  $("scores").innerHTML = `
    <table class="data">
      <thead><tr>
        <th>Lead</th><th>Event</th><th>Forecasts</th><th>Events</th>
        <th>Brier</th><th>Persistence</th><th>Climatology</th>
        <th>Skill vs persistence</th><th>Skill vs climatology</th>
      </tr></thead>
      <tbody>${rows.map((r) => `
        <tr>
          <td>${r.horizon_h} h</td><td>Kp ≥ ${r.threshold}</td><td>${r.n}</td><td>${r.events}</td>
          <td>${num(r.brier)}</td><td>${num(r.brier_persistence)}</td><td>${num(r.brier_climatology)}</td>
          <td>${signed(r.bss_persistence)}</td><td>${signed(r.bss_climatology)}</td>
        </tr>`).join("")}
      </tbody>
    </table>`;
}

function renderJobs(jobs) {
  const names = { log_forecast: "Forecast logging (every 15 min)", verify: "Observed Kp (hourly)" };
  const items = Object.entries(jobs).map(([k, j]) =>
    `${names[k] ?? k}: ${j.ok ? "ok" : "failed"} at ${dt.format(new Date(j.time))}${j.ok ? "" : ` (${esc(j.detail)})`}`);
  $("jobs").innerHTML = items.length ? `<p>${items.join("<br>")}</p>` : "<p>Background jobs are not running on this server.</p>";
}

let data;

function renderCharts() {
  if (!data) return;
  const x1 = Date.parse(data.generated);
  const x0 = x1 - data.series.days * 86400e3;
  observedChart($("chart-observed"), data.series.observed, x0, x1);
  forecastChart($("chart-forecast"), data.series.forecast_h3_kp4, x0, x1);
  brierChart($("chart-brier"), data.daily_brier);
}

async function load() {
  try {
    const r = await fetch("api/monitoring");
    if (!r.ok) throw new Error((await r.json()).detail ?? `HTTP ${r.status}`);
    data = await r.json();
  } catch (err) {
    $("updated").textContent = "monitoring data unavailable";
    $("scores").innerHTML = `<p class="unavailable">Could not load monitoring data: ${esc(err.message)}</p>`;
    return;
  }
  $("updated").textContent = `Updated ${dt.format(new Date(data.generated))}`;
  renderFacts(data);
  renderScores(data.scores);
  renderCharts();
  renderJobs(data.jobs);
}

let resizeTimer;
window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(renderCharts, 150); });
load();
setInterval(() => { if (!document.hidden) load(); }, 15 * 60 * 1000);
