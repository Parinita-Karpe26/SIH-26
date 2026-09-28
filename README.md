## Dashboard walkthrough

### What's behind it

The dashboard runs on two files: a trained model (`models/bust_detector.pkl`) and a
dataset (`data/processed_features_real.parquet`). The dataset is built from real
NCMRWF IMDAA reanalysis for July–August 2019: daily maximum temperature, daily rainfall
and 850 hPa winds, averaged onto a ~0.5° grid over India.

IMDAA tells us what the weather actually did, but it doesn't contain forecasts. So for
now we check a simple reference forecast, *persistence* ("the weather on Day N will
look like today"), which is the standard baseline in forecast verification. A forecast
counts as a **bust** if it misses maximum temperature by more than 5 °C or rainfall by
more than 50 mm. The next stage swaps persistence for NCMRWF's own S2S model forecasts;
that pipeline is already built.

When the app starts, it loads the model and data once, scores every grid point for every
forecast cycle and lead day, and keeps the results in memory. Everything you click after
that just picks a different slice. Nothing is retrained while you use it.

### Header

Shows which model is loaded, which forecast cycle you're viewing, where the data comes
from, and the server time. It's for orientation only; none of it reacts to the controls.

### Sidebar: forecast controls

- **Synoptic regime filter.** Narrows the list of forecast cycles to one type of weather
  situation, such as an active monsoon trough or a low over the Bay of Bengal. Only
  situations that actually occur in the data are listed. The labels come from a simple
  rule (rainfall over central India and circulation over the Bay), not an official
  IMD classification.
- **Forecast cycle (issue date).** Chooses which day the forecast was issued. Each entry
  shows how many lead days can be checked and the weather situation that day. Cycles near
  the end of the dataset have fewer lead days, because we can only check a forecast
  against days we have observations for. The dashboard opens on a cycle with the full
  Day 1–10 range.
- **Lead time.** Chooses how many days ahead you're looking. The slider only offers lead
  days that exist for the chosen cycle.

### Sidebar: alerting

- **Alert threshold.** The probability above which a grid point counts as a likely bust.
  It starts at the model's own operating threshold, the value that gave the best balance
  of hits and false alarms on validation days. It affects the high-risk area, alert
  regions and dominant-driver cards. The map colours don't change, because the map
  always shows the underlying probability.

### Sidebar: map display

- **Layer style.** *Heatmap* draws a smooth colour surface. *Grid points* draws individual
  points you can hover over for exact values (a sample, to keep the browser responsive).
- **Colour scale.** *Adaptive* stretches green-to-red across whatever range of risk exists
  in the current view, and the legend shows the exact range used. This matters at short
  lead times, where risk is low almost everywhere and a fixed scale would look uniformly
  green. *Fixed* always uses 0–1, so colours mean the same thing across lead days.

### Scenario banner

A one-line summary of what you're looking at: when the forecast was issued, which day
it's valid for, the weather situation, the lead day, and how many grid points inside
India are included.

### Summary cards

- **High-risk area.** The share of India's grid points at or above the alert threshold.
- **Average system confidence.** The average of (1 − bust probability) × 100 across India.
- **Alert regions.** How many of the five zones (North, South, East, West, Central) have a
  meaningful share of flagged points.
- **Dominant risk driver.** The most common reason, according to SHAP, behind the flagged
  points, for example "Sharp temperature swing". It's worked out live from the grid
  you're viewing, not a fixed label.

### The map

The risk surface is clipped to India's boundary (`app/assets/india_boundary.geojson`), so
nothing spills over neighbouring countries or the sea. Click anywhere and the inspector
below jumps to the nearest grid point.

### Drill-down and explanation

- **Region selector.** Opens on the region with the lowest average confidence, so you
  start where things look worst. It then shows that region's single highest-risk point,
  which is the one a duty officer would actually want to look at.
- **Waterfall / bar chart.** Two views of the same explanation for that point. The
  waterfall builds up from the model's average prediction to this point's prediction,
  one factor at a time. The bar chart ranks factors by impact: red pushes risk up, green
  pushes it down.
- **Meteorologist guidance summary.** A plain-English sentence built from the top
  factors, for example: *"Confidence degraded to 18% primarily due to a sharp 1-day swing
  in maximum temperature and strong 850 hPa convergence, during an active monsoon
  trough over the Indo-Gangetic Plain."* It changes with every point you select.

### Confidence by lead time

A chart of how average confidence falls as lead time grows, one line per zone, for the
selected cycle. A dashed line marks the lead day you're currently viewing.

### Good to know

- **Treat confidence as a ranking.** The probabilities rank risk well, but they aren't
  calibrated yet: the model leans towards flagging risk, so a point shown at 60% busts
  less often than 60% of the time. Use the numbers to compare places and lead days, not
  as exact odds.
- **This is a prototype,** not an official MoES/NCMRWF product.

### Suggested demo

1. Open the first cycle at **Day 1**: the map is mostly calm and confidence is high.
2. Drag the lead time towards **Day 9–10**: risk builds up across the map and the
   confidence chart falls.
3. Switch the **forecast cycle** or **regime filter** to show a different weather
   situation.
4. Click a **red area** on the map to bring up its explanation live.

### Running it

    uvicorn app.api:app --reload        # API (takes ~30 s to start)
    streamlit run app/dashboard.py      # dashboard, in a second terminal
