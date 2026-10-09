// Aurora Forecast dashboard: map, location, and the four panels.
// No build step: plain ES module + Leaflet from a CDN.

const DEFAULT = { lat: 53.55, lon: 9.99, name: "Hamburg" };
const REFRESH_MS = 10 * 60 * 1000;
const TZ = Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
const FEEDS = [
  ["solar_wind", "Solar wind"],
  ["kp", "Kp"],
  ["noaa_3day", "NOAA 3-day"],
  ["noaa_27day", "27-day"],
  ["clouds", "Clouds"],
];

const $ = (id) => document.getElementById(id);
// Leaflet needs literal colours, so read them from the CSS tokens.
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const COLORS = { bg: cssVar("--bg"), accent: cssVar("--accent"), accent2: cssVar("--accent-2") };

// ---------- formatting ----------

// Same rule as aurora/briefing.py so the briefing and the panels agree.
function pct(p) {
  if (p === null || p === undefined || Number.isNaN(p)) return "–";
  if (p < 0.01) return "<1%";
  return `${Math.round(p * 100)}%`;
}

function pctRange(lo, hi) {
  const a = pct(lo), b = pct(hi);
  if (a === b) return a;
  if (a === "<1%") return `up to ${b}`;
  return `${a.replace("%", "")}–${b}`;
}

const timeFmt = new Intl.DateTimeFormat(undefined, { hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
const dayFmt = new Intl.DateTimeFormat(undefined, { weekday: "short", day: "numeric", month: "short" });
const tzName = new Intl.DateTimeFormat(undefined, { timeZoneName: "short" });

const clock = (iso) => timeFmt.format(new Date(iso));
const day = (iso) => dayFmt.format(new Date(iso));
const tzAbbr = (iso) =>
  tzName.formatToParts(new Date(iso)).find((p) => p.type === "timeZoneName")?.value ?? "";
const kp = (v) => (v === null || v === undefined ? "–" : v.toFixed(1));

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

// NOAA G-scale badge, only for storm levels (Kp 5- counts as 5).
function stormBadge(k) {
  if (k === null || k === undefined) return "";
  const level = Math.round(k);
  if (level < 5) return "";
  const g = Math.min(level - 4, 5);
  return `<span class="storm ${level >= 7 ? "g3" : "g1"}" title="NOAA storm level">G${g}</span>`;
}

function skyLabel(h) {
  if (h.darkness >= 1) return "Dark";
  if (h.darkness > 0) return "Dusk";
  return "Day";
}

// ---------- location state ----------

function initialLocation() {
  const q = new URLSearchParams(location.search);
  const lat = parseFloat(q.get("lat")), lon = parseFloat(q.get("lon"));
  if (Number.isFinite(lat) && Number.isFinite(lon)) return { lat, lon, name: q.get("name") };
  return { ...DEFAULT };
}

let current = initialLocation();
let requestId = 0;

function setLocation(loc, { pan = true } = {}) {
  current = { lat: +loc.lat.toFixed(4), lon: +loc.lon.toFixed(4), name: loc.name || null };
  const q = new URLSearchParams({ lat: current.lat, lon: current.lon });
  if (current.name) q.set("name", current.name);
  history.replaceState(null, "", `?${q}`);
  marker.setLatLng([current.lat, current.lon]);
  if (pan) map.setView([current.lat, current.lon], Math.max(map.getZoom(), 5));
  load();
}

// ---------- map ----------

const map = L.map("map", { zoomControl: false, worldCopyJump: true, minZoom: 2 })
  .setView([current.lat, current.lon], 5);
L.control.zoom({ position: "bottomright" }).addTo(map);

// Esri Dark Gray Canvas: base and labels are separate layers, no API key needed.
// (CARTO Dark Matter now requires a key.) Tinted to the navy theme in app.css.
const ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas";
L.tileLayer(`${ESRI}/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}`, {
  attribution: "Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap contributors",
  maxZoom: 16,
  className: "tiles-base",
}).addTo(map);
L.tileLayer(`${ESRI}/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}`, {
  maxZoom: 16,
  className: "tiles-labels",
}).addTo(map);

const marker = L.circleMarker([current.lat, current.lon], {
  radius: 7,
  color: COLORS.bg,
  weight: 2,
  fillColor: COLORS.accent,
  fillOpacity: 1,
}).addTo(map);

// View lines: north of each line the aurora may be seen low on the northern horizon.
const linesLayer = L.layerGroup().addTo(map);
const LINE_LON = 16; // where the labels sit (central Europe / Scandinavia)

function labelPoint(points) {
  return points.reduce((a, b) => (Math.abs(b[1] - LINE_LON) < Math.abs(a[1] - LINE_LON) ? b : a));
}

function renderLines(data) {
  linesLayer.clearLayers();
  const legend = [];
  for (const line of data?.lines ?? []) {
    if (!line.points.length) continue;
    if (line.id === "now") {
      // Soft glow under a crisp line: the one bold element on the map.
      L.polyline(line.points, { color: COLORS.accent, weight: 14, opacity: 0.12, interactive: false }).addTo(linesLayer);
      L.polyline(line.points, { color: COLORS.accent, weight: 2, opacity: 0.95, interactive: false })
        .bindTooltip(`Kp ${kp(line.kp)} now`, { permanent: true, direction: "top", className: "line-label", offset: [0, -4] })
        .addTo(linesLayer)
        .openTooltip(labelPoint(line.points));
      legend.push(`<li><i class="lk now"></i>Visible north of this line now (Kp ${kp(line.kp)})</li>`);
    } else {
      L.polyline(line.points, { color: COLORS.accent2, weight: 1.5, opacity: 0.9, dashArray: "6 6", interactive: false })
        .bindTooltip(`Kp ${kp(line.kp)}`, { permanent: true, direction: "bottom", className: "line-label dim", offset: [0, 4] })
        .addTo(linesLayer)
        .openTooltip(labelPoint(line.points));
      legend.push(`<li><i class="lk fc"></i>${pct(line.probability)} chance of reaching this line within ${line.hours} h (Kp ${kp(line.kp)})</li>`);
    }
  }
  $("map-legend").innerHTML = legend.join("");
  $("map-legend").hidden = !legend.length;
}

map.on("click", (e) => setLocation({ lat: e.latlng.lat, lon: e.latlng.wrap().lng }, { pan: false }));

// ---------- search ----------

const input = $("search-input");
const results = $("search-results");
let searchTimer;
let found = [];

input.addEventListener("input", () => {
  clearTimeout(searchTimer);
  const q = input.value.trim();
  if (q.length < 2) { results.hidden = true; return; }
  searchTimer = setTimeout(() => search(q), 250);
});

async function search(q) {
  const url = `https://geocoding-api.open-meteo.com/v1/search?count=6&language=en&format=json&name=${encodeURIComponent(q)}`;
  try {
    const data = await (await fetch(url)).json();
    found = data.results || [];
  } catch {
    found = [];
  }
  results.innerHTML = found.length
    ? found.map((r, i) =>
        `<li role="option" data-i="${i}">${esc(r.name)}<small>${esc([r.admin1, r.country].filter(Boolean).join(", "))}</small></li>`).join("")
    : `<li aria-disabled="true">No places found</li>`;
  results.hidden = false;
}

function pick(i) {
  const r = found[i];
  if (!r) return;
  results.hidden = true;
  input.value = "";
  input.blur();
  setLocation({ lat: r.latitude, lon: r.longitude, name: r.name });
}

results.addEventListener("click", (e) => {
  const li = e.target.closest("li[data-i]");
  if (li) pick(+li.dataset.i);
});

$("search").addEventListener("submit", (e) => { e.preventDefault(); if (found.length) pick(0); });
input.addEventListener("keydown", (e) => { if (e.key === "Escape") results.hidden = true; });
document.addEventListener("click", (e) => { if (!$("search").contains(e.target)) results.hidden = true; });

$("locate").addEventListener("click", () => {
  if (!navigator.geolocation) return;
  navigator.geolocation.getCurrentPosition(
    (p) => setLocation({ lat: p.coords.latitude, lon: p.coords.longitude, name: "My location" }),
    () => { $("updated").textContent = "location access denied"; },
    { timeout: 10000 },
  );
});

// ---------- loading ----------

async function getJSON(path, params) {
  const r = await fetch(`${path}?${new URLSearchParams(params)}`);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json();
}

async function load() {
  const id = ++requestId;
  const loc = { ...current };
  $("updated").textContent = "updating…";
  let fc;
  try {
    fc = await getJSON("api/forecast", { lat: loc.lat, lon: loc.lon });
  } catch (err) {
    if (id !== requestId) return;
    $("updated").textContent = "forecast unavailable";
    for (const p of ["now", "nights", "weeks"]) {
      $(`${p}-body`).innerHTML = `<p class="unavailable">Could not reach the forecast service. Retrying in a few minutes.</p>`;
    }
    console.error(err);
    return;
  }
  // Fetched after the forecast so both share its cached global inputs.
  const lines = await getJSON("api/view-lines", {}).catch(() => null);
  if (id !== requestId) return;
  renderStatus(fc);
  renderChip(fc, loc);
  renderLines(lines);
  renderNow(fc, lines);
  renderNights(fc);
  renderWeeks(fc);
  loadBriefing(id, loc);
}

async function loadBriefing(id, loc) {
  $("briefing-text").innerHTML = `<span class="skeleton"></span><span class="skeleton short"></span>`;
  $("briefing-tag").textContent = "";
  try {
    const params = { lat: loc.lat, lon: loc.lon, tz: TZ };
    if (loc.name) params.place = loc.name;
    const b = await getJSON("api/briefing", params);
    if (id !== requestId) return;
    $("briefing-text").textContent = b.text;
    $("briefing-tag").textContent = b.source.startsWith("llm") ? "AI-written · Claude Haiku 4.5" : "Template";
    $("briefing-tag").title = b.source.startsWith("llm")
      ? "Phrased by an LLM from the numbers below; replies with numbers not in the forecast are rejected."
      : "LLM not configured or unavailable, so a fixed template is used.";
  } catch {
    if (id !== requestId) return;
    $("briefing-text").textContent = "Briefing unavailable. The panels below are not affected.";
  }
}

// ---------- render: status + chip ----------

function renderStatus(fc) {
  $("feeds").innerHTML = FEEDS.map(([key, label]) => {
    const down = key in fc.errors;
    return `<li class="${down ? "down" : ""}" title="${down ? "unavailable" : "ok"}">${label}${down ? " · down" : ""}</li>`;
  }).join("");
  $("updated").textContent = `Updated ${clock(fc.generated)} ${tzAbbr(fc.generated)}`;
}

function renderChip(fc, loc) {
  const l = fc.location;
  const ns = l.lat >= 0 ? "N" : "S", ew = l.lon >= 0 ? "E" : "W";
  const coords = `${Math.abs(l.lat).toFixed(2)}°${ns}, ${Math.abs(l.lon).toFixed(2)}°${ew}`;
  $("chip-place").textContent = loc.name ? `${loc.name}` : coords;
  $("chip-place").title = coords;
  $("chip-mlat").textContent = `${l.mlat.toFixed(1)}°`;
  $("chip-kp").textContent = l.kp_needed > 9 ? ">9" : kp(l.kp_needed);
  const notes = [];
  if (l.hemisphere === "south") notes.push("Southern lights: look low on the southern horizon. The map lines show the northern hemisphere only.");
  if (!l.tuned_region) notes.push("Kp thresholds are tuned for Europe and are less accurate here.");
  $("chip-note").textContent = notes.join(" ");
  $("chip-note").hidden = notes.length === 0;
  $("map-chip").hidden = false;
}

function unavailable(source) {
  return `<p class="unavailable">${esc(source)} is unavailable right now. The other panels still work.</p>`;
}

// ---------- render: now ----------

function renderNow(fc, lines) {
  const body = $("now-body"), foot = $("now-foot");
  const now = fc.now;
  if (!now || !now.hours.length) {
    body.innerHTML = unavailable("Real-time solar wind data");
    foot.textContent = "";
    return;
  }
  const hours = now.hours;
  const hasClouds = hours.every((h) => h.chance !== null);
  const key = hasClouds ? "chance" : "chance_if_clear";
  const best = hours.reduce((a, b) => (b[key] > a[key] ? b : a));
  const kpNeeded = fc.location.kp_needed;
  const kpLast = now.kp_forecast.kp_last;
  const fcLine = lines?.lines?.find((l) => l.id === "forecast");

  // Bar scale: a clean ceiling so tiny values stay tiny instead of filling the slot.
  const top = Math.max(...hours.map((h) => h.chance_if_clear ?? 0));
  const scale = [0.02, 0.05, 0.1, 0.2, 0.5, 1].find((s) => top <= s);
  const h = (v) => `${Math.max(0, Math.min(1, (v ?? 0) / scale)) * 100}%`;

  const cols = hours.map((r) => {
    const tip = [
      `${clock(r.time)}`,
      `chance ${pct(r.chance)}`,
      `if clear ${pct(r.chance_if_clear)}`,
      `P(Kp ≥ ${kp(kpNeeded)}) ${pct(r.p_kp)} (${r.horizon_h} h model, ${r.confidence} confidence)`,
      `clouds ${pct(r.cloud_cover)}`,
      `darkness ${Math.round(r.darkness * 100)}%`,
      `moon ${pct(r.moon_illumination)} lit`,
    ].join("\n");
    return { r, tip };
  });

  body.innerHTML = `
    <div class="hero-row">
      <div>
        <div class="hero-value num">${pct(best[key])}</div>
        <div class="hero-label">${hasClouds ? "Best chance" : "Best chance if clear"}, at ${clock(best.time)}</div>
      </div>
      ${kpScale(kpLast, kpNeeded, fcLine)}
    </div>

    <div class="hours" style="--n:${hours.length}" role="table" aria-label="Next hours">
      <div class="rowlabel">scale 0–${pct(scale)}</div>
      ${cols.map(({ r, tip }) => `
        <div class="bar-slot" title="${esc(tip)}">
          <div class="if-clear" style="height:${h(r.chance_if_clear)}"></div>
          <div class="bar" style="height:${h(r.chance)}"></div>
        </div>`).join("")}
      <div class="rowlabel">Time</div>
      ${hours.map((r) => `<div class="cell time num">${clock(r.time)}</div>`).join("")}
      <div class="rowlabel">Chance</div>
      ${hours.map((r) => `<div class="cell value num">${pct(r.chance)}</div>`).join("")}
      <div class="rowlabel">If clear</div>
      ${hours.map((r) => `<div class="cell num">${pct(r.chance_if_clear)}</div>`).join("")}
      <div class="rowlabel" title="Probability that geomagnetic activity reaches the Kp needed here">Kp ≥ ${kpNeeded > 9 ? "9" : kp(kpNeeded)}</div>
      ${hours.map((r) => `<div class="cell num">${pct(r.p_kp)}</div>`).join("")}
      <div class="rowlabel">Clouds</div>
      ${hours.map((r) => `<div class="cell num">${pct(r.cloud_cover)}</div>`).join("")}
      <div class="rowlabel">Sky</div>
      ${hours.map((r) => `<div class="cell">${skyLabel(r)}</div>`).join("")}
    </div>
    <div class="legend">
      <span><i class="key solid"></i>Chance</span>
      <span><i class="key outline"></i>Chance if clear</span>
    </div>

    <details>
      <summary>Kp model output</summary>
      <table class="data">
        <thead><tr><th>Lead</th><th>Kp interval</th><th>P(Kp≥5)</th><th>P(Kp≥6)</th><th>P(Kp≥7)</th></tr></thead>
        <tbody>
          ${now.kp_forecast.horizons.map((hz) => `
            <tr>
              <td>${hz.horizon_h} h</td>
              <td>${clock(hz.interval_start)}–${clock(hz.interval_end)}</td>
              <td>${pct(hz.p_kp_at_least["5"])}</td>
              <td>${pct(hz.p_kp_at_least["6"])}</td>
              <td>${pct(hz.p_kp_at_least["7"])}</td>
            </tr>`).join("")}
        </tbody>
      </table>
      ${now.solar_wind ? `
      <dl class="kv">
        <dt>Spacecraft</dt><dd>${esc((now.solar_wind.spacecraft || []).join(", "))}</dd>
        <dt>Latest L1 sample</dt><dd>${clock(now.solar_wind.latest_l1_sample)}</dd>
        <dt>Distance to Earth</dt><dd>${(now.solar_wind.l1_distance_km / 1e6).toFixed(2)} million km</dd>
        <dt>Travel time to Earth</dt><dd>${Math.round(now.solar_wind.propagation_minutes)} min</dd>
        <dt>Forecast issued for</dt><dd>${clock(now.kp_forecast.issued)}</dd>
      </dl>` : ""}
    </details>`;

  foot.textContent = `Source: ${now.source}.`;
}

// Kp 0–9 scale: where activity is, where the forecast may take it, and what this place needs.
function kpScale(now, needed, fcLine) {
  const x = (k) => `${(Math.max(0, Math.min(9, k)) / 9) * 100}%`;
  const ticks = Array.from({ length: 10 }, (_, k) =>
    `<span class="tick${k >= 7 ? " g3" : k >= 5 ? " g1" : ""}" style="left:${x(k)}">${k}</span>`).join("");
  const reach = fcLine && fcLine.kp > now
    ? `<div class="reach" style="left:${x(now)};width:calc(${x(fcLine.kp)} - ${x(now)})" title="${pct(fcLine.probability)} chance of reaching Kp ${kp(fcLine.kp)} within ${fcLine.hours} h"></div>`
    : "";
  const neededMark = needed <= 9
    ? `<div class="mark needed" style="left:${x(needed)}"><span>needed here ${kp(needed)}</span></div>`
    : `<div class="mark needed beyond"><span>needed here: above 9</span></div>`;
  return `
    <figure class="kpscale" aria-label="Kp now ${kp(now)}, needed here ${needed > 9 ? "above 9" : kp(needed)}">
      <figcaption>Geomagnetic activity (Kp) ${stormBadge(now)}</figcaption>
      <div class="track">
        <div class="storm-zone" style="left:${x(5)}"></div>
        ${reach}
        <div class="mark now" style="left:${x(now)}"><span>now ${kp(now)}</span></div>
        ${neededMark}
      </div>
      <div class="ticks">${ticks}</div>
      ${fcLine && fcLine.kp > now ? `<p class="reach-note"><i></i>${pct(fcLine.probability)} chance of Kp ${kp(fcLine.kp)} within ${fcLine.hours} h</p>` : ""}
    </figure>`;
}

// ---------- render: nights ----------

function renderNights(fc) {
  const body = $("nights-body"), foot = $("nights-foot");
  const panel = fc.nights;
  if (!panel || !panel.nights.length) {
    body.innerHTML = unavailable("The 3-night forecast");
    foot.textContent = "";
    return;
  }
  const nights = panel.nights;
  // Shared scale across the three nights so their ranges compare.
  const top = Math.max(...nights.map((n) => (n.chance_range ? n.chance_range[1] : n.chance_if_clear) ?? 0));
  const scale = [0.02, 0.05, 0.1, 0.2, 0.5, 1].find((s) => top <= s);
  const x = (v) => Math.max(0, Math.min(100, (v / scale) * 100));

  body.innerHTML = `<div class="nights-grid">${nights.map((n, i) => {
    const name = i === 0 ? "Tonight" : day(n.start);
    const hasRange = n.chance_range !== null;
    const [lo, hi] = hasRange ? n.chance_range : [0, n.chance_if_clear];
    const src = n.sources.map((s) => (s === "model" ? "Model" : "NOAA")).join(" + ");
    return `
      <div class="night">
        <div class="night-name">${name}</div>
        <div class="night-window num">${clock(n.start)}–${clock(n.end)}</div>
        ${hasRange
          ? `<div class="night-value num">${pctRange(lo, hi)}</div>
             <div class="night-sub">chance, best around ${clock(n.best_hour)}</div>`
          : `<div class="night-value if-clear num">≤ ${pct(n.chance_if_clear)}</div>
             <div class="night-sub">if clear · no cloud forecast yet</div>`}
        <div class="range-track" title="0–${pct(scale)} scale"><div class="range-fill" style="left:${x(lo)}%;width:${Math.max(x(hi) - x(lo), 0)}%"></div></div>
        <dl class="kv">
          <dt>If clear</dt><dd>${pct(n.chance_if_clear)}</dd>
          <dt>Clouds</dt><dd>${n.cloud_cover_mean === null ? "–" : pct(n.cloud_cover_mean)}</dd>
          <dt>Moon</dt><dd>${pct(n.moon_illumination)} lit</dd>
          <dt>NOAA Kp max</dt><dd>${kp(n.noaa_kp_max)} ${stormBadge(n.noaa_kp_max)}</dd>
        </dl>
        <span class="src">${src}</span>
      </div>`;
  }).join("")}</div>`;
  foot.textContent = `Ranges show the uncertainty of NOAA's forecast calibration. Source: ${panel.source}.`;
}

// ---------- render: weeks ----------

function renderWeeks(fc) {
  const body = $("weeks-body"), foot = $("weeks-foot");
  const w = fc.weeks;
  if (!w) {
    body.innerHTML = unavailable("The 27-day outlook");
    foot.textContent = "";
    return;
  }
  const until = w.outlook_end ? day(`${w.outlook_end}T12:00:00Z`) : "the end of the outlook";
  const days = w.possible_activity;
  if (!days.length) {
    body.innerHTML = `<p class="weeks-text">No elevated activity expected through ${until}.</p>`;
  } else {
    body.innerHTML = `<ul class="weeks-list">${days.map((d) => {
      const date = day(`${d.date}T12:00:00Z`);
      const rec = d.kp_27_days_ago !== null ? `; Kp ${kp(d.kp_27_days_ago)} one solar rotation ago` : "";
      const here = d.reaches_kp_needed ? " — could reach the level needed here" : "";
      return `<li>Possible activity around <strong>${date}</strong> (NOAA outlook Kp ${d.outlook_kp_max}${rec})${here}.</li>`;
    }).join("")}</ul>`;
  }
  foot.textContent =
    "Based on NOAA's 27-day outlook and the 27-day solar rotation. Recurring coronal-hole streams can be anticipated; CME-driven storms cannot.";
}

// ---------- go ----------

load();
setInterval(() => { if (!document.hidden) load(); }, REFRESH_MS);
