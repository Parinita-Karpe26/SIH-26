Glad it's rendering correctly now — that's a proper India outline with the Northeast (Assam/Arunachal) states also showing color, which is right since they're genuinely part of the dataset. Here's a full walkthrough of everything on the page, what it does, and what's actually happening behind it.

## The big picture

Everything you see is driven by one pre-trained model (`bust_detector.pkl`) and one dataset (`processed_features.parquet`). On startup, the app loads both **once**, runs the model across the whole dataset to get a `bust_probability` for every grid point/lead-day, and caches that in memory. Every widget you touch afterward just filters or re-slices that cached result — nothing gets retrained or reloaded as you interact.

## Header bar

Static branding plus three live status pills: which model type is loaded (XGBoost), which forecast cycle is "current" (the most recent date in the dataset), and the server's current time. This is cosmetic/status only — it doesn't respond to your controls.

## Sidebar — Forecast Controls

- **Lead Time slider (Day 1–10):** picks which of the 10 forecast lead times to display. Moving it re-filters the cached predictions to that lead day — no recomputation of the model itself, just a different slice.
- **Synoptic Event Filter:** this is the more interesting one. It doesn't filter *within* the current forecast — it **switches which archived forecast case you're looking at entirely**. Our synthetic dataset has 36 different forecast dates spread across a year, each dominated by one weather pattern (monsoon trough, western disturbance, Bay of Bengal cyclone, or quiescent). Picking "Monsoon Depression" finds the most recent archived case tagged as a Bay-of-Bengal cyclone and shows *that* case's full Day 1–10 forecast instead. "Heat Wave" is flagged in the UI as a **derived proxy** (quiescent regime + top-10% temperature) since our generator never modeled a true heat-wave pattern — that's disclosed honestly in the "About" panel rather than hidden.

## Sidebar — Alerting

- **Alert Threshold slider:** this is the probability cutoff for "is this officially a bust risk or not." It only affects two things: the "High-Risk Area %" KPI and the "Alert Regions" KPI. It does **not** change the map's colors — the map always shows the raw continuous probability, not a pass/fail flag.

## Sidebar — Map Display

- **Layer style (Heatmap / Grid Points):** Heatmap renders a smooth, continuous color surface (technically a rasterized image, not individual blobs — this is the fix from a couple of messages ago). Grid Points instead draws individual dots you can hover over for exact numbers, sampled to keep the browser fast.
- **Colour scale (Adaptive / Fixed):** Adaptive stretches green→red to whatever range of risk actually exists *in the view you're currently looking at* — this is why the legend right above the map literally prints the numbers it's using (e.g., "0.001 → 0.003" in your screenshot). Without this, Day 1–5 forecasts would look uniformly flat green, since real risk at short lead times is genuinely tiny. Fixed instead always uses the literal 0.00–1.00 probability scale, so colors mean the same thing across every lead day — useful if you want strict comparability rather than "what's relatively worse right now."

## Scenario banner

The blue strip confirming exactly what you're looking at: which date this forecast was issued, what weather regime dominates it, which lead day, and how many actual India grid points are included (not the full rectangle — see below).

## KPI cards

- **High-Risk Area (% of India):** the share of India's grid cells whose probability is at or above your Alert Threshold.
- **Avg. System Confidence:** the mean of `(1 − bust_probability) × 100` across all of India for this view.
- **Alert Regions:** how many of the 5 macro-zones (North/South/East/West/Central India) have a meaningful chunk of flagged high-risk points.
- **Dominant Risk Driver:** the single most common SHAP-identified cause of risk across all flagged points right now (e.g., "Extended lead time," "Active low-pressure system"). This comes from actually running SHAP over the visible grid, not a hardcoded label.

## The map itself

The color field is now clipped to India's real political boundary (a locally-shipped GeoJSON), not the old rectangular data box — that was the fix for the Pakistan/China/ocean bleed you flagged. You can click anywhere on the map; if a browser click registers, it snaps to the nearest real India grid point and feeds it into the drill-down section below instead of whatever the region dropdown had selected.

## Drill-Down & XAI Inspector

- **Region dropdown:** defaults to whichever region currently has the *lowest* average confidence — i.e., it auto-points you at the most concerning area without you having to hunt for it. Picking a region shows its single highest-risk grid point (not an average — the point a duty officer would actually want to see).
- **Waterfall / Bar Chart tabs:** both show the same underlying SHAP explanation for that one point, just two visual styles. The waterfall shows how you get from the model's average baseline probability up (or down) to this specific point's prediction, feature by feature. The bar chart just ranks the same features by impact, red = pushes risk up, green = pushes it down.
- **Meteorologist Guidance Summary:** a plain-English sentence generated from those same SHAP values (e.g., "Confidence degraded to 12% primarily due to an active low-pressure system near the Bay of Bengal coastal belt..."). This is template-composed from the top 1–2 real contributing features, not a canned string — it changes based on whatever point you're actually looking at.

## Lead-Time Confidence Decay chart

A Plotly line chart showing how average confidence falls off from Day 1 to Day 10, one line per macro-zone, for whichever scenario is currently selected. The dashed vertical line marks whatever lead day your slider is currently on, so you can see where "today's" view sits on that decay curve.

## Footer

Just the synthetic-data disclaimer, restated for anyone who scrolls straight to the bottom.

A good way to demo this to judges: start on "Live Forecast Cycle" at Day 1 (flat, high confidence), drag the lead slider to Day 10 (watch the map develop texture and the decay chart drop), then switch the Event Filter to "Monsoon Depression" to show it's a genuinely different archived storm, and finish by clicking a red patch on the map to pull up its SHAP explanation live.
