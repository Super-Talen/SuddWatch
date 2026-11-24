/* SuddWatch public page: everything shown comes from the API (/forecast/latest, /model) and static files. */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const pct = (p) => `${Math.round(p * 100)}%`;
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const fmtDate = (iso, opts) => {
    const [y, m, d] = iso.split("-").map(Number);
    return `${d} ${MONTHS[m - 1]}${opts && opts.year ? " " + y : ""}`;
  };
  const fmtRange = (a, b) => (a.slice(0, 7) === b.slice(0, 7) ? `${Number(a.slice(8))}–${fmtDate(b)}` : `${fmtDate(a)} – ${fmtDate(b)}`);
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const getJSON = (url) => fetch(url).then((r) => { if (!r.ok) throw new Error(url + " " + r.status); return r.json(); });

  let threshold = 0.5, bins = [], map, layer, byCode = {}, layersByCode = {}, selected = null;

  function makeBins(t) {
    const edges = [0, 0.2, t, 0.6, 0.8, 1.0001];
    return edges.slice(0, -1).map((lo, i) => ({ lo, hi: edges[i + 1], color: `--p${i}`, alert: lo >= t }));
  }
  const binOf = (p) => bins.find((b) => p >= b.lo && p < b.hi) || bins[bins.length - 1];
  const colorOf = (p) => css(binOf(p).color);

  /* ---------- hero ---------- */
  function hero(fc) {
    const top = fc.forecasts[0];
    $("t-alerts").textContent = `${fc.counties_on_alert} of ${fc.forecasts.length}`;
    $("t-top").textContent = pct(top.probability);
    $("t-top-label").textContent = `highest risk: ${top.county}`;
    $("t-window").textContent = fmtRange(fc.forecast_window_start, fc.forecast_window_end);
    $("hero-lede").textContent = `Forecast issued ${fmtDate(fc.issue_date, { year: "numeric" })} from the latest satellite pass. ` +
      `It covers ${fmtDate(fc.forecast_window_start)} to ${fmtDate(fc.forecast_window_end, { year: "numeric" })}, 13 to 24 days ahead.`;
    $("f-window").textContent = `${fmtDate(fc.forecast_window_start)} and ${fmtDate(fc.forecast_window_end, { year: "numeric" })}`;
  }

  /* ---------- map ---------- */
  function styleFor(code) {
    const f = byCode[code];
    const state = $("state").value;
    const dim = state && f && f.state !== state;
    return {
      fillColor: f ? colorOf(f.probability) : css("--hairline"),
      fillOpacity: dim ? 0.18 : 0.9,
      color: code === selected ? css("--ink") : css("--county-line"),
      weight: code === selected ? 2.5 : 1,
    };
  }

  function tipHTML(f) {
    const div = el("div");
    div.appendChild(el("strong", null, f.county));
    div.appendChild(el("div", "tip-state", f.state));
    const p = el("div", null, `Flood chance: ${pct(f.probability)}`);
    div.appendChild(p);
    div.appendChild(el("div", null, f.alert ? "▲ Alert: above county normal expected" : "No alert"));
    return div;
  }

  function drawMap(geo) {
    // No background tiles: the county shapes carry the map, and the page needs no third-party tile service.
    map = L.map("map", { scrollWheelZoom: false, zoomSnap: 0.25, attributionControl: false, minZoom: 5, maxZoom: 9 });
    layer = L.geoJSON(geo, {
      style: (feat) => styleFor(feat.properties.adm2_pcode),
      onEachFeature: (feat, lyr) => {
        const code = feat.properties.adm2_pcode;
        layersByCode[code] = lyr;
        const f = byCode[code];
        if (f) lyr.bindTooltip(tipHTML(f), { sticky: true, className: "county-tip", direction: "top", offset: [0, -8] });
        lyr.on({
          mouseover: () => { if (code !== selected) lyr.setStyle({ weight: 2, color: css("--ink-2") }); lyr.bringToFront(); },
          mouseout: () => { lyr.setStyle(styleFor(code)); if (selected && layersByCode[selected]) layersByCode[selected].bringToFront(); },
          click: () => select(code, false),
        });
      },
    }).addTo(map);
    map.fitBounds(layer.getBounds(), { padding: [8, 8] });
    addEventListener("resize", () => map.invalidateSize());
  }

  function legend() {
    const box = $("legend");
    box.textContent = "";
    box.appendChild(el("span", null, "Flood chance"));
    const scale = el("div", "legend-scale");
    bins.forEach((b) => {
      const step = el("div", "legend-step" + (b.alert ? " alert" : ""));
      const sw = el("i"); sw.style.background = css(b.color);
      step.appendChild(sw);
      step.appendChild(el("span", null, `${Math.round(b.lo * 100)}–${Math.min(100, Math.round(b.hi * 100))}%`));
      scale.appendChild(step);
    });
    box.appendChild(scale);
    box.appendChild(el("span", null, `Alert from ${pct(threshold)}`));
  }

  /* ---------- list ---------- */
  function drawList(forecasts) {
    const state = $("state").value;
    const rows = forecasts.filter((f) => !state || f.state === state);
    const list = $("county-list");
    list.textContent = "";
    rows.forEach((f) => {
      const li = el("li");
      li.tabIndex = 0;
      li.dataset.code = f.county_pcode;
      if (f.county_pcode === selected) li.className = "selected";
      const name = el("div", "c-name", f.county);
      if (f.alert) name.appendChild(el("span", "pill", "▲ ALERT"));
      name.appendChild(el("span", "c-state", f.state));
      const bar = el("div", "c-bar"); const fill = el("i");
      fill.style.width = pct(f.probability); fill.style.background = colorOf(f.probability);
      bar.appendChild(fill);
      li.append(name, bar, el("div", "c-pct", pct(f.probability)));
      li.addEventListener("click", () => select(f.county_pcode, true));
      li.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(f.county_pcode, true); } });
      list.appendChild(li);
    });
    const on = rows.filter((f) => f.alert).length;
    $("list-count").textContent = `${on} of ${rows.length} on alert`;
  }

  function select(code, fromList) {
    const prev = selected;
    selected = code === selected ? null : code;
    [prev, selected].forEach((c) => c && layersByCode[c] && layersByCode[c].setStyle(styleFor(c)));
    document.querySelectorAll("#county-list li").forEach((li) => li.classList.toggle("selected", li.dataset.code === selected));
    if (!selected) return;
    const lyr = layersByCode[selected];
    if (lyr) {
      lyr.bringToFront();
      if (fromList) { map.flyToBounds(lyr.getBounds(), { maxZoom: 8, duration: 0.6 }); lyr.openTooltip(lyr.getBounds().getCenter()); }
    }
    if (!fromList) {
      const li = document.querySelector(`#county-list li[data-code="${selected}"]`);
      if (li) li.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }

  /* ---------- model scores ---------- */
  function scores(card) {
    const t = card.test_metrics;
    const items = [
      [t.pr_auc.toFixed(2), "ranking skill (PR-AUC, 1 = perfect)"],
      [t.macro_f1_dynamic.toFixed(2), "balanced accuracy of alerts (Macro-F1)"],
      [pct(t.false_alarm_dynamic), "of non-flood cases wrongly alerted"],
      [t.ece_dynamic.toFixed(3), "calibration error (0 = probabilities exact)"],
    ];
    const box = $("scores");
    box.textContent = "";
    items.forEach(([v, label]) => { const s = el("div", "score"); s.append(el("b", null, v), el("span", null, label)); box.appendChild(s); });
    const p = el("p", "note", `Scores on the held-out test period (July 2025 – August 2026, ${t.rows.toLocaleString()} county forecasts).`);
    box.after(p);
  }

  /* ---------- photos ---------- */
  function figure(c, extra) {
    const fig = el("figure");
    const img = el("img"); img.src = c.src; img.alt = c.title; img.loading = "lazy";
    img.width = c.width; img.height = c.height;
    const cap = el("figcaption", null, extra || c.title);
    const small = el("small");
    small.append("Photo: ", c.author, " · ");
    const a = el("a", null, c.license); a.href = c.license_url; a.rel = "license noopener"; a.target = "_blank";
    small.append(a, " · ");
    const s = el("a", null, "source"); s.href = c.source_url; s.target = "_blank"; s.rel = "noopener";
    small.appendChild(s);
    cap.appendChild(small);
    fig.append(img, cap);
    return fig;
  }

  function photos(credits) {
    const by = Object.fromEntries(credits.map((c) => [c.key, c]));
    const h = by.jonglei_children;
    if (h) {
      const img = $("hero-img"); img.src = h.src; img.alt = h.title; img.hidden = false;
      const cr = $("hero-credit");
      cr.append(`${h.title}. Photo: ${h.author}, `);
      const a = el("a", null, h.license); a.href = h.license_url; a.target = "_blank"; a.rel = "license noopener";
      cr.appendChild(a); cr.hidden = false;
    }
    const why = ["maban_flooding", "health_facility"].filter((k) => by[k]);
    if (why.length) { why.forEach((k) => $("why-photos").appendChild(figure(by[k]))); $("why").hidden = false; }
    const ev = ["bentiu_before_after", "bentiu_from_space"].filter((k) => by[k]);
    if (ev.length) {
      const text = {
        bentiu_before_after: "The same area around Bentiu on 21 November 2020 (left) and 26 November 2021 (right). In this false-colour image, floodwater shows as dark blue: a year later it covers much of the land south and west of the town.",
        bentiu_from_space: "A wider false-colour view of Bentiu and the wetlands and river channels around it, published by ESA in 2023.",
      };
      ev.forEach((k) => $("evidence-photos").appendChild(figure(by[k], text[k])));
      $("evidence").hidden = false;
    }
    if (credits.length) {
      const f = $("credits");
      f.appendChild(el("p", null, "Photographs and satellite images, used under their licenses (not modified apart from resizing):"));
      const ul = el("ul");
      credits.forEach((c) => {
        const li = el("li"); li.append(`${c.title}: ${c.author}, `);
        const a = el("a", null, c.license); a.href = c.license_url; a.target = "_blank"; a.rel = "license noopener";
        li.append(a, ", via ");
        const s = el("a", null, "Wikimedia Commons"); s.href = c.source_url; s.target = "_blank"; s.rel = "noopener";
        li.appendChild(s); ul.appendChild(li);
      });
      f.appendChild(ul);
    }
  }

  /* ---------- boot ---------- */
  Promise.all([getJSON("/forecast/latest"), getJSON("/model"), getJSON("/static/data/counties.geojson")])
    .then(([fc, card, geo]) => {
      threshold = card.decision_threshold;
      bins = makeBins(threshold);
      $("f-threshold").textContent = pct(threshold);
      fc.forecasts.forEach((f) => { byCode[f.county_pcode] = f; });
      hero(fc);
      drawMap(geo);
      legend();
      [...new Set(fc.forecasts.map((f) => f.state))].sort().forEach((s) => $("state").appendChild(new Option(s, s)));
      $("state").addEventListener("change", () => {
        drawList(fc.forecasts);
        Object.keys(layersByCode).forEach((c) => layersByCode[c].setStyle(styleFor(c)));
        const st = $("state").value;
        if (st) {
          const b = L.latLngBounds([]);
          fc.forecasts.filter((f) => f.state === st).forEach((f) => layersByCode[f.county_pcode] && b.extend(layersByCode[f.county_pcode].getBounds()));
          if (b.isValid()) map.flyToBounds(b, { padding: [20, 20], duration: 0.6 });
        } else map.flyToBounds(layer.getBounds(), { padding: [8, 8], duration: 0.6 });
      });
      drawList(fc.forecasts);
      scores(card);
    })
    .catch((err) => {
      $("hero-lede").textContent = "The forecast could not be loaded. Please try again in a moment.";
      console.error(err);
    });

  getJSON("/static/img/credits.json").then(photos).catch(() => {});
})();
