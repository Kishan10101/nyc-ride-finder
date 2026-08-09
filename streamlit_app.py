"""
NYC Ride Finder — a road-cycling ride planner for New York.

Pick a start point, pick a route, and see what you are riding into: 7-day wind and
weather, live precipitation radar, heart-rate zones from your own numbers, and
fueling math that keys off intensity rather than distance.

Run locally:
    pip install -r requirements.txt
    streamlit run streamlit_app.py

Personal details are read from Streamlit secrets or environment variables so a
public deployment need not contain a street address. See README.
"""

from __future__ import annotations

import json
import math
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pydeck as pdk
import streamlit as st

st.set_page_config(
    page_title="NYC Ride Finder",
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _setting(key: str, default):
    """Config from Streamlit secrets, then environment, then default."""
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


# ============================================================================
# START POINTS
# ============================================================================

@dataclass(frozen=True)
class StartPoint:
    key: str
    label: str
    address: str
    lat: float
    lon: float
    note: str


HOME_ADDRESS = str(_setting("HOME_ADDRESS", "Fort Greene, Brooklyn, NY 11205"))
HOME_LAT = float(_setting("HOME_LAT", 40.6893))
HOME_LON = float(_setting("HOME_LON", -73.9737))

STARTS: Dict[str, StartPoint] = {
    "ftgreene": StartPoint(
        "ftgreene", "Fort Greene", HOME_ADDRESS, HOME_LAT, HOME_LON,
        "Brooklyn side. Closest to Prospect Park and everything south.",
    ),
    "barclays": StartPoint(
        "barclays", "Barclays HQ · Midtown", "745 7th Ave, New York, NY 10019",
        40.7614, -73.9832,
        "745 Seventh Ave, the old Lehman Brothers building. Two blocks off the "
        "Central Park south entrance and a short run to the Hudson greenway.",
    ),
    "exchange": StartPoint(
        "exchange", "Exchange Place · Jersey City", "Exchange Place, Jersey City, NJ 07302",
        40.7163, -74.0330,
        "Jersey side. Liberty State Park is minutes away; anything in Brooklyn "
        "means a PATH ride or a long detour.",
    ),
}

# Reference point the stored route distances were measured from
BASE_START = STARTS["ftgreene"]

# Detour factor: straight-line distance understates road distance in a street grid
ROAD_FACTOR = 1.28


# ============================================================================
# GEOMETRY
# ============================================================================

def haversine_mi(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    r = 3958.7613
    lat1, lon1, lat2, lon2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def bearing_deg(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lat2 = math.radians(a[0]), math.radians(b[0])
    dlon = math.radians(b[1] - a[1])
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def compass_point(deg: float) -> str:
    pts = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return pts[int((deg % 360) / 22.5 + 0.5) % 16]


def wind_components(wind_from_deg: float, wind_mph: float, heading_deg: float) -> Tuple[float, float]:
    """
    Return (headwind, crosswind) in mph for a rider on `heading_deg`.
    wind_from_deg is the direction the wind blows FROM, as meteorology reports it.
    Positive headwind means it is against you.
    """
    delta = math.radians(wind_from_deg - heading_deg)
    return wind_mph * math.cos(delta), abs(wind_mph * math.sin(delta))


# ============================================================================
# HEART RATE MODEL
# ============================================================================

ZONE_NAMES = ["Z1 · Recovery", "Z2 · Endurance", "Z3 · Tempo",
              "Z4 · Threshold", "Z5 · VO2max"]

ZONE_FEEL = {
    "Z1 · Recovery": "Barely working. Could hold this all day.",
    "Z2 · Endurance": "Full conversation possible. The bread and butter.",
    "Z3 · Tempo": "Short sentences only. Comfortably hard.",
    "Z4 · Threshold": "A few words at a time. Sustainable 20–60 min.",
    "Z5 · VO2max": "No talking. Minutes, not hours.",
}

ZONES_HRMAX = {
    "Z1 · Recovery": (0.50, 0.60), "Z2 · Endurance": (0.60, 0.70),
    "Z3 · Tempo": (0.70, 0.80), "Z4 · Threshold": (0.80, 0.90),
    "Z5 · VO2max": (0.90, 1.00),
}
ZONES_LTHR = {
    "Z1 · Recovery": (0.65, 0.81), "Z2 · Endurance": (0.81, 0.89),
    "Z3 · Tempo": (0.90, 0.93), "Z4 · Threshold": (0.94, 0.99),
    "Z5 · VO2max": (1.00, 1.06),
}

# Heart rate reserve edges — the method Apple Watch uses on Automatic
HRR_BOUNDS = (0.59, 0.69, 0.783, 0.89)

# Carbohydrate share of energy expenditure vs % heart rate reserve.
# %HRR tracks %VO2max; fat oxidation peaks near 45–65% and carbs take over above.
_CARB_ANCHORS = [(0.30, 0.30), (0.40, 0.35), (0.50, 0.42), (0.60, 0.55),
                 (0.70, 0.68), (0.80, 0.80), (0.90, 0.90), (1.00, 0.97)]


def carb_fraction_from_hrr(pct: float) -> float:
    pct = max(0.20, min(1.05, pct))
    if pct <= _CARB_ANCHORS[0][0]:
        return _CARB_ANCHORS[0][1]
    if pct >= _CARB_ANCHORS[-1][0]:
        return _CARB_ANCHORS[-1][1]
    for (x0, y0), (x1, y1) in zip(_CARB_ANCHORS, _CARB_ANCHORS[1:]):
        if x0 <= pct <= x1:
            return y0 + (pct - x0) / (x1 - x0) * (y1 - y0)
    return _CARB_ANCHORS[-1][1]


def tanaka_max_hr(age: int) -> int:
    return int(round(208 - 0.7 * age))


def hrr_zone_edges(max_hr: int, rest_hr: int) -> List[int]:
    hrr = max(1, max_hr - rest_hr)
    return [int(rest_hr + b * hrr) for b in HRR_BOUNDS]


def build_zones(method: str, max_hr: int, rest_hr: int, lthr: int,
                edges: List[int]) -> Dict[str, Tuple[int, int]]:
    if method == "% of max HR":
        return {n: (int(round(max_hr * lo)), int(round(max_hr * hi)))
                for n, (lo, hi) in ZONES_HRMAX.items()}
    if method == "% of LTHR":
        return {n: (int(round(lthr * lo)), int(round(lthr * hi)))
                for n, (lo, hi) in ZONES_LTHR.items()}
    e = sorted(edges)
    floor = int(rest_hr + 0.30 * max(1, max_hr - rest_hr))
    return {
        ZONE_NAMES[0]: (floor, e[0]),
        ZONE_NAMES[1]: (e[0] + 1, e[1]),
        ZONE_NAMES[2]: (e[1] + 1, e[2]),
        ZONE_NAMES[3]: (e[2] + 1, e[3]),
        ZONE_NAMES[4]: (e[3] + 1, max_hr),
    }


def pct_hrr(hr: float, max_hr: int, rest_hr: int) -> float:
    return (hr - rest_hr) / max(1, max_hr - rest_hr)


def keytel_kcal_hr(hr: float, kg: float, age: int, sex: str) -> float:
    if sex == "Male":
        kj = -55.0969 + 0.6309 * hr + 0.1988 * kg + 0.2017 * age
    else:
        kj = -20.4022 + 0.4472 * hr - 0.1263 * kg + 0.0740 * age
    return max(0.0, kj / 4.184) * 60


def vo2_kcal_hr(pct: float, vo2max: float, kg: float, carb_frac: float) -> float:
    vo2 = 3.5 + pct * max(0.0, vo2max - 3.5)          # ml/kg/min
    litres = vo2 * kg / 1000.0
    return litres * (4.69 + 0.36 * carb_frac) * 60     # kcal/L varies with substrate


def fueling_model(dur_hr: float, zlo: int, zhi: int, max_hr: int, rest_hr: int,
                  kg: float, age: int, sex: str, mixed: bool,
                  vo2max: Optional[float]) -> dict:
    hr = (zlo + zhi) / 2.0
    pct = pct_hrr(hr, max_hr, rest_hr)
    cf = carb_fraction_from_hrr(pct)
    k1 = keytel_kcal_hr(hr, kg, age, sex)
    k2 = vo2_kcal_hr(pct, vo2max, kg, cf) if vo2max else None
    kcal = (k1 + k2) / 2 if k2 else k1
    ox = kcal * cf / 4.0
    ceiling = 90 if mixed else 60
    intake = min(ox * 0.60, ceiling)
    if dur_hr < 1.25:
        intake = 0.0 if pct < 0.70 else min(intake, 25)
    fuel_hours = max(0.0, dur_hr - 0.5)
    fl = max(0.3, 0.35 + 0.75 * pct)
    return dict(hr=int(round(hr)), pct_hrr=pct, pct_max=hr / max(1, max_hr),
                kcal_keytel=k1, kcal_vo2=k2, kcal_hr=kcal, kcal_total=kcal * dur_hr,
                carb_frac=cf, ox=ox, intake=intake, total_intake=intake * fuel_hours,
                ceiling=ceiling, fluid_hr=fl, fluid_total=fl * dur_hr,
                post_carb=kcal * dur_hr * 0.30 / 4.0,
                post_protein=min(40, max(20, kg * 0.35)))


def carb_examples(g: float) -> str:
    g = int(round(g))
    if g <= 5:
        return "—"
    return (f"{max(1, round(g / 22))} gels · {max(1, round(g / 27))} bananas · "
            f"{max(1, round(g / 18))} dates · {max(1, round(g / 40))} bottles of mix")


# ============================================================================
# WEATHER — Open-Meteo, no API key required
# ============================================================================

OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
RAINVIEWER_INDEX = "https://api.rainviewer.com/public/weather-maps.json"


def _get_json(url: str, timeout: int = 12) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "nyc-ride-finder/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_forecast(lat: float, lon: float, days: int = 7) -> dict:
    """7-day hourly forecast. Returns {'ok': bool, 'hourly': DataFrame|None, 'error': str}."""
    q = urllib.parse.urlencode({
        "latitude": round(lat, 4), "longitude": round(lon, 4),
        "hourly": ",".join([
            "temperature_2m", "apparent_temperature", "precipitation_probability",
            "precipitation", "wind_speed_10m", "wind_direction_10m",
            "wind_gusts_10m", "cloud_cover", "uv_index", "is_day",
        ]),
        "daily": "sunrise,sunset,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
        "precipitation_unit": "inch", "timezone": "auto",
        "forecast_days": days,
    })
    try:
        raw = _get_json(f"{OPEN_METEO}?{q}")
        h = raw["hourly"]
        df = pd.DataFrame({
            "time": pd.to_datetime(h["time"]),
            "temp": h["temperature_2m"],
            "feels": h["apparent_temperature"],
            "precip_prob": h["precipitation_probability"],
            "precip": h["precipitation"],
            "wind": h["wind_speed_10m"],
            "wind_dir": h["wind_direction_10m"],
            "gust": h["wind_gusts_10m"],
            "cloud": h["cloud_cover"],
            "uv": h["uv_index"],
            "is_day": h["is_day"],
        })
        return {"ok": True, "hourly": df, "daily": raw.get("daily", {}), "error": ""}
    except Exception as exc:
        return {"ok": False, "hourly": None, "daily": {}, "error": f"{type(exc).__name__}: {exc}"}


@st.cache_data(ttl=600, show_spinner=False)
def fetch_radar_frame() -> Optional[str]:
    """Latest RainViewer radar tile template, or None if unavailable."""
    try:
        idx = _get_json(RAINVIEWER_INDEX)
        past = idx.get("radar", {}).get("past", [])
        if not past:
            return None
        path = past[-1]["path"]
        host = idx.get("host", "https://tilecache.rainviewer.com")
        return f"{host}{path}/256/{{z}}/{{x}}/{{y}}/4/1_1.png"
    except Exception:
        return None


IDEAL_TEMP_F = 64.0


def ride_score(row) -> float:
    """
    Score an hour 0–100 for road riding. Deliberately harsh on rain and gusts,
    because those are the two things that actually ruin or endanger a ride.
    """
    s = 100.0
    s -= min(60.0, row["precip_prob"] * 0.60)            # 100% chance → −60
    s -= min(25.0, row["precip"] * 220.0)                # measurable rain → heavy hit
    s -= min(30.0, max(0.0, row["wind"] - 8.0) * 1.9)    # calm under 8 mph
    s -= min(20.0, max(0.0, row["gust"] - 18.0) * 1.5)   # gusts are the dangerous part
    s -= min(25.0, abs(row["temp"] - IDEAL_TEMP_F) * 0.85)
    if row["temp"] < 38 or row["temp"] > 92:
        s -= 15.0
    if not row["is_day"]:
        s -= 30.0
    return max(0.0, min(100.0, s))



# ---------------------------------------------------------------------------
# PLAIN LANGUAGE — the app has to make sense to someone who has never heard
# of a heart rate zone, without dumbing anything down for someone who has.
# ---------------------------------------------------------------------------

def wind_in_words(mph: float) -> str:
    if mph < 4:
        return "Calm. You will not notice it."
    if mph < 8:
        return "Light breeze. Pleasant."
    if mph < 13:
        return "Noticeable. You will feel it heading into it."
    if mph < 19:
        return "Strong. Hard work one way, easy the other."
    if mph < 25:
        return "Very strong. Pick a sheltered route or a shorter day."
    return "Too strong to enjoy. Consider indoors."


def temp_in_words(f: float) -> str:
    if f < 32:
        return "Freezing. Watch for ice."
    if f < 45:
        return "Cold. Full gloves, long sleeves, cover your ears."
    if f < 58:
        return "Cool. Arm warmers or a light jacket."
    if f < 75:
        return "Ideal riding weather."
    if f < 86:
        return "Warm. Carry extra water."
    return "Hot. Ride early, drink more than feels necessary."


def rain_in_words(pct: float, inches: float) -> str:
    if pct < 15:
        return "Dry."
    if pct < 40:
        return "Small chance of rain. Probably fine."
    if inches > 0.05 or pct >= 70:
        return "Rain likely. Roads will be slick."
    return "Rain possible. Take a jacket."


def flatness(r) -> str:
    per_mi = r.elevation_ft / max(1.0, r.base_total)
    if per_mi < 8:
        return "Flat"
    if per_mi < 22:
        return "Gently rolling"
    return "Hilly"


def traffic_comfort(r) -> str:
    if r.car_free and r.interruption_score <= 2:
        return "Very relaxed — no cars"
    if r.car_free:
        return "Relaxed — car-free, but shared with people on foot"
    if r.interruption_score >= 4:
        return "Busy — city streets and traffic lights"
    return "Moderate — some road riding"


_TREAT_WORDS = ("coffee", "café", "cafe", "roasting", "bakery", "taco", "food",
                "bbq", "rapha", "devoción", "devocion", "sey", "bunbury",
                "runcible", "brancaccio", "grocer", "deli")


def has_treat(r) -> bool:
    blob = " ".join((s.name + " " + s.what).lower() for s in r.stops)
    return any(w in blob for w in _TREAT_WORDS)


def beginner_ok(r) -> bool:
    # A ferry is not a barrier for a new rider — Governors Island is a family
    # day out. Distance, climbing and traffic are what actually gate people.
    return (r.base_total <= 30 and r.elevation_ft <= 420
            and r.interruption_score <= 4)


def score_label(s: float) -> str:
    if s >= 85:
        return "Excellent"
    if s >= 70:
        return "Good"
    if s >= 55:
        return "Workable"
    if s >= 38:
        return "Poor"
    return "Stay in"


# ============================================================================
# DESIGN SYSTEM
# ============================================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500&display=swap');

:root {
  --ink:   #101418;
  --paper: #FCFCFA;
  --rule:  #D6DAD4;
  --route: #0E7C86;
  --wind:  #E8460F;
  --muted: #6B7580;
}

html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; color: var(--ink); }
.stApp { background: var(--paper); }
.block-container { padding-top: 2.2rem; max-width: 1180px; }

h1 { font-family:'Archivo',sans-serif !important; font-weight:700 !important;
     letter-spacing:-0.035em !important; font-size:2.1rem !important; margin-bottom:0 !important; }
h2, h3 { font-family:'Archivo',sans-serif !important; font-weight:600 !important;
         letter-spacing:-0.02em !important; }

/* Section heads are a rule and a word, not a badge. */
.sec {
  border-top: 2px solid var(--ink);
  margin: 1.9rem 0 0.7rem;
  padding-top: 0.4rem;
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.7rem; font-weight: 600;
  letter-spacing: 0.13em; text-transform: uppercase; color: var(--ink);
}
.sec span { color: var(--muted); font-weight: 400; letter-spacing: 0.06em; }

/* Data strip: a spec-sheet row. No boxes, no fills, no shadows. */
.strip { display:flex; flex-wrap:wrap; border-top:1px solid var(--ink);
         border-bottom:1px solid var(--rule); margin:0.2rem 0 0.9rem; }
.strip .cell { flex:1 1 0; min-width:104px; padding:0.6rem 0.9rem 0.55rem 0;
               border-right:1px solid var(--rule); }
.strip .cell:last-child { border-right:none; }
.strip .num { font-family:'IBM Plex Mono',monospace; font-variant-numeric:tabular-nums;
              font-size:1.32rem; font-weight:600; letter-spacing:-0.02em; line-height:1.1; }
.strip .cap { font-family:'IBM Plex Mono',monospace; font-size:0.6rem; font-weight:500;
              letter-spacing:0.12em; text-transform:uppercase; color:var(--muted);
              margin-top:0.28rem; }

/* Key/value rows, like a cue sheet. */
.rowline { display:flex; justify-content:space-between; gap:1.2rem;
           padding:0.36rem 0; border-bottom:1px solid var(--rule); font-size:0.87rem; }
.rowline:first-child { border-top:1px solid var(--rule); }
.rowline:last-child { border-bottom:1px solid var(--rule); }
.rowline .k { color:var(--muted); }
.rowline .v { font-family:'IBM Plex Mono',monospace; font-variant-numeric:tabular-nums;
              text-align:right; }

/* Dense forecast table. */
.tt { width:100%; border-collapse:collapse; font-family:'IBM Plex Mono',monospace;
      font-size:0.79rem; font-variant-numeric:tabular-nums; }
.tt th { text-align:left; font-weight:600; font-size:0.6rem; letter-spacing:0.12em;
         text-transform:uppercase; color:var(--muted); padding:0.3rem 0.7rem 0.3rem 0;
         border-bottom:1px solid var(--ink); white-space:nowrap; }
.tt td { padding:0.42rem 0.7rem 0.42rem 0; border-bottom:1px solid var(--rule);
         white-space:nowrap; }
.tt tr:hover td { background:#F2F4F0; }
.tt .day { font-weight:600; }
.bar { display:inline-block; height:7px; vertical-align:middle; }
.bartrack { display:inline-block; width:78px; height:7px; background:#E7E9E4;
            vertical-align:middle; margin-right:0.5rem; }

.lede { font-size:1.02rem; line-height:1.6; color:var(--ink); max-width:62ch; }
.note { font-size:0.83rem; color:var(--muted); line-height:1.55; max-width:68ch; }
.flag { border-left:2px solid var(--wind); padding:0.15rem 0 0.15rem 0.8rem;
        font-size:0.9rem; line-height:1.55; margin:0.6rem 0 0.9rem; }
.flag--ok { border-left-color:var(--route); }

.pill { font-family:'IBM Plex Mono',monospace; font-size:0.62rem; font-weight:500;
        letter-spacing:0.08em; text-transform:uppercase; color:var(--muted);
        margin-right:0.9rem; }
.pill--on { color:var(--route); }

.stTabs [data-baseweb="tab-list"] { gap:1.6rem; border-bottom:1px solid var(--ink); }
.stTabs [data-baseweb="tab"] { font-family:'IBM Plex Mono',monospace; font-size:0.72rem;
  font-weight:600; letter-spacing:0.11em; text-transform:uppercase; color:var(--muted);
  background:transparent !important; padding:0.35rem 0; }
.stTabs [aria-selected="true"] { color:var(--ink) !important; }
.stTabs [data-baseweb="tab-highlight"] { background:var(--ink); }

section[data-testid="stSidebar"] { background:#F5F6F2; border-right:1px solid var(--rule); }
[data-testid="stMetric"] { background:transparent; border:none; padding:0; }
.stMetric [data-testid="stMetricValue"] { font-family:'IBM Plex Mono',monospace !important;
  font-variant-numeric:tabular-nums; font-weight:600 !important; font-size:1.3rem !important; }
[data-testid="stMetricLabel"] { font-family:'IBM Plex Mono',monospace !important;
  font-size:0.6rem !important; letter-spacing:0.12em !important; text-transform:uppercase;
  color:var(--muted) !important; }
hr { border-color: var(--rule) !important; }

*:focus-visible { outline:3px solid var(--wind) !important; outline-offset:2px !important; }
.stApp a { color:#0A5C64; text-underline-offset:2px; }
@media (max-width:760px) {
  .strip .cell { flex:1 1 46%; border-right:none; border-bottom:1px solid var(--rule); }
  .strip .num { font-size:1.1rem; }
  .rowline { flex-direction:column; gap:0.1rem; }
  .rowline .v { text-align:left; }
  .block-container { padding-top:1.2rem; }
}
@media (prefers-reduced-motion: reduce) { * { animation:none !important; transition:none !important; } }
</style>
"""


def eyebrow(text: str, sub: str = "") -> None:
    """A section head is a rule and a word. Not a badge, not a card title."""
    tail = f" <span>{sub}</span>" if sub else ""
    st.markdown(f"<div class='sec'>{text}{tail}</div>", unsafe_allow_html=True)


def datastrip(items: List[Tuple[str, str]]) -> None:
    """Numbers laid out like a spec sheet: hairline columns, no tiles."""
    cells = "".join(
        f"<div class='cell'><div class='num'>{v}</div>"
        f"<div class='cap'>{k}</div></div>" for k, v in items
    )
    st.markdown(f"<div class='strip'>{cells}</div>", unsafe_allow_html=True)


def bar_cell(pct: float, color: str) -> str:
    w = max(2, min(78, int(pct / 100 * 78)))
    return (f"<span class='bartrack'><span class='bar' style='width:{w}px;"
            f"background:{color}'></span></span>")


def rows(pairs: List[Tuple[str, str]]) -> str:
    return "".join(
        f"<div class='rowline'><span class='k'>{k}</span>"
        f"<span class='v'>{v}</span></div>" for k, v in pairs
    )


def wind_rose_svg(route_bearing: float, wind_from: float, wind_mph: float,
                  gust_mph: float, size: int = 210) -> str:
    """
    The signature element. A bearing dial showing where the route points and where
    the wind is pushing, with the headwind component resolved numerically.
    """
    c = size / 2
    r = c - 26
    head, cross = wind_components(wind_from, wind_mph, route_bearing)

    def pt(deg: float, rad: float) -> Tuple[float, float]:
        a = math.radians(deg - 90)
        return c + rad * math.cos(a), c + rad * math.sin(a)

    ticks = []
    for d in range(0, 360, 15):
        major = d % 45 == 0
        x1, y1 = pt(d, r)
        x2, y2 = pt(d, r - (9 if major else 5))
        ticks.append(
            f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' "
            f"stroke='#C9CFC7' stroke-width='{1.4 if major else 0.8}'/>"
        )
    labels = []
    for d, t in [(0, "N"), (90, "E"), (180, "S"), (270, "W")]:
        x, y = pt(d, r + 13)
        labels.append(
            f"<text x='{x:.1f}' y='{y + 4:.1f}' text-anchor='middle' "
            f"font-family='IBM Plex Mono, monospace' font-size='10' "
            f"font-weight='600' fill='#5A6570'>{t}</text>"
        )

    # Route heading — teal, from centre outward
    rx, ry = pt(route_bearing, r - 16)
    # Wind vector — orange, drawn pointing the way the wind blows TO
    blow_to = (wind_from + 180) % 360
    wx, wy = pt(blow_to, r - 16)
    wtail_x, wtail_y = pt(wind_from, r - 16)

    verdict = ("HEADWIND" if head > 1.5 else "TAILWIND" if head < -1.5 else "CROSSWIND")
    vcolor = "#FF5A1F" if head > 1.5 else "#0E7C86" if head < -1.5 else "#5A6570"

    return f"""
<svg viewBox="0 0 {size} {size + 58}" width="100%" style="max-width:{size + 40}px"
     role="img" aria-label="Wind relative to route bearing">
  <circle cx="{c}" cy="{c}" r="{r}" fill="#FFFFFF" stroke="#C9CFC7" stroke-width="1"/>
  {''.join(ticks)}{''.join(labels)}
  <defs>
    <marker id="mr" markerWidth="7" markerHeight="7" refX="5.4" refY="3"
            orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="#0E7C86"/></marker>
    <marker id="mw" markerWidth="7" markerHeight="7" refX="5.4" refY="3"
            orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="#FF5A1F"/></marker>
  </defs>
  <line x1="{c}" y1="{c}" x2="{rx:.1f}" y2="{ry:.1f}" stroke="#0E7C86"
        stroke-width="2.6" marker-end="url(#mr)"/>
  <line x1="{wtail_x:.1f}" y1="{wtail_y:.1f}" x2="{wx:.1f}" y2="{wy:.1f}"
        stroke="#FF5A1F" stroke-width="2.2" stroke-dasharray="5 3"
        marker-end="url(#mw)"/>
  <circle cx="{c}" cy="{c}" r="3.4" fill="#141A1F"/>
  <text x="{c}" y="{size + 16}" text-anchor="middle"
        font-family="IBM Plex Mono, monospace" font-size="19" font-weight="600"
        fill="{vcolor}">{abs(head):.0f} mph</text>
  <text x="{c}" y="{size + 32}" text-anchor="middle"
        font-family="IBM Plex Mono, monospace" font-size="9" font-weight="600"
        letter-spacing="1.6" fill="{vcolor}">{verdict} OUTBOUND</text>
  <text x="{c}" y="{size + 50}" text-anchor="middle"
        font-family="IBM Plex Mono, monospace" font-size="9"
        fill="#5A6570">{cross:.0f} mph cross · gust {gust_mph:.0f} · route {route_bearing:.0f}°</text>
</svg>"""


# ============================================================================
# ROUTES
# ============================================================================

@dataclass
class Stop:
    name: str
    lat: float
    lon: float
    what: str


@dataclass
class Route:
    name: str
    blurb: str
    base_total: float                  # door-to-door miles from Fort Greene
    base_approach: float               # one-way approach from Fort Greene
    ride_miles: str                    # the riding once you are there
    elevation_ft: int
    surface: str
    interruptions: str
    interruption_score: int            # 1 = open road, 5 = signal every block
    car_free: bool
    ferry_needed: bool
    exposure: str                      # how sheltered from wind
    best_for: List[str]
    difficulty: str
    typical_zone: str
    stats: List[Tuple[str, str]]
    approach: Dict[str, str]           # start key -> one-line approach note
    on_route: List[str]                # start-independent directions
    description: str
    tips: List[str]
    stops: List[Stop]
    warnings: List[str]
    waypoints: List[Tuple[float, float]]
    gmaps_dest: str
    is_loop: bool = False
    gmaps_via: List[str] = field(default_factory=list)

    def entry(self) -> Tuple[float, float]:
        return self.waypoints[0]

    def far_point(self) -> Tuple[float, float]:
        e = self.entry()
        return max(self.waypoints, key=lambda w: haversine_mi(e, w))

    def approach_from(self, sp: StartPoint) -> float:
        return haversine_mi((sp.lat, sp.lon), self.entry()) * ROAD_FACTOR

    def total_from(self, sp: StartPoint) -> float:
        a = self.approach_from(sp)
        if self.is_loop:
            return self.base_total + 2 * a
        core = max(0.0, self.base_total - 2 * self.base_approach)
        return core + 2 * a

    def heading_from(self, sp: StartPoint) -> float:
        return bearing_deg((sp.lat, sp.lon), self.far_point())

    def gmaps_url(self, sp: StartPoint) -> str:
        p = {"api": "1", "origin": sp.address, "destination": self.gmaps_dest,
             "travelmode": "bicycling"}
        if self.gmaps_via:
            p["waypoints"] = "|".join(self.gmaps_via)
        return "https://www.google.com/maps/dir/?" + urllib.parse.urlencode(p)


R = Route
S = Stop

ROUTES: List[Route] = [

    R(name="Prospect Park Loop",
      blurb="Car-free laps ten minutes from Fort Greene. One real hill.",
      base_total=17.0, base_approach=1.9, ride_miles="3.35 mi per lap",
      elevation_ft=131, surface="Smooth asphalt, dedicated bike lane",
      interruptions="No cars, but shared with runners and walkers",
      interruption_score=2, car_free=True, ferry_needed=False,
      exposure="Sheltered — mature trees the whole loop",
      best_for=["Endurance", "Tempo", "Weeknight ride", "Beginner-friendly"],
      difficulty="Easy", typical_zone="Z2 · Endurance",
      stats=[("Loop", "3.35 mi (Prospect Park Alliance)"),
             ("Elevation per lap", "+131 ft"),
             ("Direction", "Counter-clockwise, one-way"),
             ("Car-free since", "Jan 2, 2018, permanently"),
             ("Park size", "585 acres"),
             ("Lap time", "11–13 min at 16–18 mph"),
             ("Riding allowed on", "Park, Center and Wellhouse Drives only")],
      approach={"ftgreene": "Vanderbilt Ave protected lane straight south to Grand Army Plaza, 1.9 mi.",
                "barclays": "Manhattan Bridge then Flatbush Ave, or take the B/Q to Grand Army Plaza with the bike.",
                "exchange": "PATH to WTC, Manhattan Bridge, then Flatbush — the Jersey start makes this an awkward one."},
      on_route=["Enter at Grand Army Plaza onto **West Drive**, bear right onto **Park Drive**.",
                "Counter-clockwise. The climb comes on the east side past the zoo.",
                "Stack laps to distance: 6 laps ≈ 20 mi and ~800 ft of climbing."],
      description=(
          "The ride you will do most, if you are starting in Brooklyn. Nothing else in the "
          "city puts you on protected car-free pavement this fast.\n\n"
          "One 3.35-mile counter-clockwise circuit, one genuine climb, and pavement among "
          "the best inside city limits. The loop went permanently car-free in January 2018 "
          "after recreational users had been outnumbering cars roughly three to one.\n\n"
          "Because it is tree-lined the whole way, it is also the most wind-sheltered "
          "option here — worth knowing on a day the forecast is ugly."),
      tips=["**Before 8am is a different park.** After 9am on a weekend the lanes fill and it "
            "becomes an exercise in brake modulation.",
            "Stay in the bike lane. It is separate from the pedestrian lane and locals enforce it.",
            "Use the Grand Army Plaza arch as start/finish for lap counting.",
            "**Best bad-weather option.** The tree cover kills most of the wind, so on a "
            "20 mph day this loses far less than the waterfront routes.",
            "Watch HR drift lap over lap — the loop is short enough that you can see fatigue "
            "separate from crowd-dodging."],
      stops=[S("Grand Army Plaza", 40.6736, -73.9700, "Soldiers' and Sailors' Arch. Start/finish and group-ride meetup."),
             S("Picnic House", 40.6690, -73.9720, "Bathrooms and water, mid-park on West Drive."),
             S("Prospect Park Lake", 40.6533, -73.9720, "Southern tip. Brooklyn's only lake."),
             S("Bicycle Habitat", 40.6800, -73.9690, "560 Vanderbilt Ave, on the Fort Greene approach.")],
      warnings=["Poor venue for hard intervals — you spend the effort dodging rather than riding."],
      waypoints=[(40.6736, -73.9700), (40.6690, -73.9720), (40.6533, -73.9720),
                 (40.6620, -73.9640), (40.6736, -73.9700)],
      gmaps_dest="Grand Army Plaza, Brooklyn, NY"),

    R(name="Central Park Loop",
      blurb="6.1 miles and ~390 ft per lap. The best hills in Manhattan.",
      base_total=25.0, base_approach=5.5, ride_miles="6.1 mi per lap",
      elevation_ft=390, surface="Smooth asphalt",
      interruptions="Low in the park, heavy on the approach from Brooklyn",
      interruption_score=3, car_free=True, ferry_needed=False,
      exposure="Sheltered — the park's tree canopy blocks most wind",
      best_for=["Hill repeats", "Tempo", "Endurance", "Group riding"],
      difficulty="Moderate", typical_zone="Z3 · Tempo",
      stats=[("Loop", "6.1 mi (9.8 km)"),
             ("Elevation per lap", "~390 ft — three times a Prospect lap"),
             ("Harlem Hill", "3.4% valley-to-crest, 4.4% over its steepest 0.32 mi"),
             ("Cat Hill", "Near E 75th, gradual, panther statue at the top"),
             ("Named climbs", "9, only Cat and Harlem exceed 5% anywhere"),
             ("Direction", "Counter-clockwise"),
             ("Bail-out", "102nd St Crossing skips Harlem Hill")],
      approach={"ftgreene": "Manhattan Bridge, Chrystie St, then the 2nd Ave protected lane north — 5.5 mi of stop-and-go.",
                "barclays": "Two blocks. You are at the door: 7th Ave north to Columbus Circle or 59th St.",
                "exchange": "PATH to 33rd St, then 6th Ave north — about 2.5 mi once you are across."},
      on_route=["Enter at **Grand Army Plaza** (5th Ave & 59th) onto **East Park Drive**.",
                "Counter-clockwise: north up the east side, Cat Hill around 75th.",
                "**Harlem Hill** is the northwest corner, 106th–110th.",
                "Full lap returns you to 59th. The 102nd St Crossing cuts it short."],
      description=(
          "The best pure loop in the city. Six miles a lap with roughly 390 feet of "
          "climbing — about three times the vertical of a Prospect lap over less than "
          "double the distance.\n\n"
          "Harlem Hill on the northwest corner is the local benchmark: 3.4% average "
          "valley-to-crest, 4.4% over its steepest third of a mile. Nine named climbs sit "
          "on the loop but only Harlem and Cat break five percent anywhere.\n\n"
          "**Which start you pick changes this ride completely.** From Barclays it is a "
          "two-block warm-up and the best loop in the city is yours. From Fort Greene the "
          "approach costs 70+ minutes of stop-and-go at 10–11 mph, which is commuting, not "
          "training."),
      tips=["**Take the Manhattan Bridge, not the Brooklyn.** The Brooklyn Bridge path is a "
            "tourist obstacle course.",
            "The 2nd Ave protected lane is the best north-south route in Manhattan.",
            "**Harlem Hill repeats:** the north end stays uncrowded even on busy weekends. "
            "If the loop is packed, do 4–6 repeats of that segment alone.",
            "Watch for pedicabs. They stop without warning and do not signal.",
            "Before 7am the loop belongs to fast group rides.",
            "A lap HR average sits between the climb spike and the descent crater and "
            "describes neither. Judge the climbs on their own."],
      stops=[S("Grand Army Plaza", 40.7644, -73.9737, "5th Ave & 59th. Standard loop start."),
             S("Harlem Hill", 40.7970, -73.9580, "106th–110th. The benchmark climb."),
             S("Cat Hill", 40.7745, -73.9660, "East Drive near 75th. The panther statue."),
             S("Engineers' Gate", 40.7820, -73.9590, "90th & 5th. Water, bathrooms, the meeting point."),
             S("Rapha New York", 40.7258, -74.0007, "159 Prince St, SoHo. Clubhouse café — a detour on the way home.")],
      warnings=["From Brooklyn the approach is not training mileage — it drags the day's average "
                "speed down 3–4 mph. Track laps separately.",
                "Red lights inside the park are ticketed, especially at the 72nd and 79th transverses."],
      waypoints=[(40.7644, -73.9737), (40.7745, -73.9660), (40.7970, -73.9580),
                 (40.7900, -73.9620), (40.7644, -73.9737)],
      gmaps_dest="Grand Army Plaza, Central Park South, New York, NY"),

    R(name="Hudson River Greenway + Rapha",
      blurb="Fourteen car-free miles, then the SoHo clubhouse.",
      base_total=24.0, base_approach=3.0, ride_miles="~14 mi of separated path",
      elevation_ft=320, surface="Separated greenway, then SoHo street",
      interruptions="Very low on the path, heavy for the last SoHo blocks",
      interruption_score=3, car_free=False, ferry_needed=False,
      exposure="Fully exposed — open river on your flank the whole way",
      best_for=["Endurance", "Tempo", "Social ride", "Group riding"],
      difficulty="Easy-Moderate", typical_zone="Z2 · Endurance",
      stats=[("Rapha New York", "159 Prince St, at Prince and West Broadway, SoHo"),
             ("Hours", "About 10am–6pm, later Thu–Fri. Verify before planning around it."),
             ("Car-free path miles", "~14 of the round trip"),
             ("What it is", "Retail plus a full café, home of the NYC Rapha Cycling Club"),
             ("Group rides", "Depart from here; the classic route heads north to Piermont"),
             ("Path character", "Fully separated the length of Manhattan's west side")],
      approach={"ftgreene": "Manhattan Bridge then Canal St west to the water, about 3 mi.",
                "barclays": "West on 57th St to the river, under a mile. This is your home path.",
                "exchange": "PATH to WTC and you surface a few blocks from the greenway — the shortest approach of any start."},
      on_route=["Join the **Hudson River Greenway** at the waterfront.",
                "**North** along the river — car-free the entire way. Clears out above 72nd St.",
                "Turn at your distance, then run it back south.",
                "Finish at **159 Prince St** — cut east a few blocks from Canal."],
      description=(
          "The single best piece of cycling infrastructure in New York, and the right "
          "structure for a café ride: do the work first, collect the reward after.\n\n"
          "The greenway runs fully separated the length of Manhattan's west side. Fourteen "
          "car-free miles means you can hold a genuine steady effort for over an hour "
          "without touching a car.\n\n"
          "You finish at Rapha New York on Prince Street — a clubhouse rather than a shop: "
          "retail up front, a proper café, live racing on screens through the season, and "
          "the home of the New York chapter of the Rapha Cycling Club. Group rides set off "
          "from here toward the rolling roads around Piermont, which makes this the most "
          "efficient way into the local road scene.\n\n"
          "**Wind matters more here than anywhere else on this list.** The path is fully "
          "exposed along the river, and a northerly means you fight it out and get pushed "
          "home. Check the rose before you commit to a direction."),
      tips=["**Ride out into the wind, come home with it.** On an exposed out-and-back this "
            "is the whole game — check the rose and pick your direction accordingly.",
            "Busy with pedestrians below 72nd St, much clearer north of there.",
            "**Confirm the clubhouse hours before you build a ride around it.**",
            "The RCC posts rides on Strava, and shop rides do not require membership.",
            "**Extend it:** continue to the GWB for a 40-mile day, or bolt on a Central Park lap.",
            "Bring a lock. You are leaving the bike on a SoHo street."],
      stops=[S("Rapha New York", 40.7258, -74.0007, "159 Prince St. Clubhouse café, retail, race screenings, RCC home."),
             S("Pier 45 / Hudson River Park", 40.7350, -74.0110, "Mid-path. Water, bathrooms, benches."),
             S("Little Island", 40.7415, -74.0100, "Pier 55. The park on stilts. Free, bikes stay outside."),
             S("Dinosaur BBQ", 40.8180, -73.9600, "131st & 12th Ave, right off the path. Longtime cyclist stop."),
             S("Little Red Lighthouse", 40.8500, -73.9470, "Under the GWB, if you run the path to its north end.")],
      warnings=["The Canal St and SoHo blocks are dense, double-parked and full of delivery "
                "traffic. Slow down for those few blocks.",
                "Retail hours and locations change. Confirm before riding out."],
      waypoints=[(40.7200, -74.0100), (40.7350, -74.0110), (40.7550, -74.0080),
                 (40.7770, -73.9900), (40.8000, -73.9750), (40.7400, -74.0100),
                 (40.7258, -74.0007)],
      gmaps_dest="Rapha New York, 159 Prince St, New York, NY",
      gmaps_via=["Hudson River Greenway, New York, NY"]),

    R(name="Liberty State Park",
      blurb="Flattest, smoothest, emptiest riding in the metro area.",
      base_total=26.0, base_approach=8.0, ride_miles="~6 mi of park paths plus waterfront",
      elevation_ft=80, surface="Smooth paths and promenade",
      interruptions="Very low inside the park",
      interruption_score=2, car_free=True, ferry_needed=True,
      exposure="Fully exposed — reclaimed flat on open harbor",
      best_for=["Intervals", "Time trial", "Tempo", "Threshold testing"],
      difficulty="Easy", typical_zone="Z4 · Threshold",
      stats=[("Park size", "1,212 acres"),
             ("Waterfront Walkway", "~2 mi car-free promenade"),
             ("Elevation", "Effectively zero — reclaimed rail yard and fill"),
             ("Ferry", "Liberty Landing carries bikes; NY Waterway also serves Jersey City"),
             ("Alternative access", "PATH to Newport or Exchange Place, then ride south"),
             ("PATH restriction", "Bikes restricted during weekday rush hours")],
      approach={"ftgreene": "Manhattan Bridge, Hudson greenway south to Battery, then ferry or PATH across.",
                "barclays": "PATH from 33rd St to Exchange Place, then ride south along the walkway.",
                "exchange": "You are already here. Ride the waterfront walkway south — under 2 mi, no ferry, no PATH."},
      on_route=["From the Jersey City waterfront, **Hudson River Waterfront Walkway** south "
                "through Paulus Hook.",
                "**Essex St**, then **Jersey Ave** south — keep straight onto the pedestrian bridge.",
                "You are in the park. The waterfront promenade runs south toward Morris Pesin Dr.",
                "Out-and-back efforts along the promenade cancel out the wind."],
      description=(
          "**If you are starting at Exchange Place, this is your home venue and it is a "
          "genuinely good one.** Under two miles of car-free walkway and you are on the "
          "flattest, smoothest, emptiest riding in the region.\n\n"
          "Liberty State Park is 1,212 acres of reclaimed rail yard: completely flat, wide "
          "smooth paths, and very little foot traffic away from the Statue viewing area. "
          "The two-mile promenade lets you hold a hard steady effort with Ellis Island and "
          "the Statue of Liberty directly ahead.\n\n"
          "For a Brooklyn start it is a production — ferry or PATH, schedules, bike "
          "restrictions. From Jersey City it is the obvious daily ride."),
      tips=["**From Exchange Place this needs no ferry at all.** Just ride south. That "
            "single fact makes it the best interval venue in the app for a Jersey start.",
            "The walkway north of the park adds 4–5 more flat car-free miles.",
            "**Best flat time-trial venue in the region.** Do clean out-and-backs to cancel wind.",
            "Fully exposed, so a windy day genuinely changes the numbers. Use the rose.",
            "Grove Street in downtown Jersey City is a serious food neighborhood now.",
            "From Brooklyn: confirm the last ferry. Getting stranded means a long detour via Bayonne."],
      stops=[S("Liberty State Park", 40.7050, -74.0550, "1,212 acres, flat car-free paths, Statue views."),
             S("Empty Sky Memorial", 40.7110, -74.0410, "New Jersey's 9/11 memorial, framing where the towers stood."),
             S("Liberty Science Center", 40.7080, -74.0560, "Bathrooms, water, café."),
             S("Paulus Hook waterfront", 40.7150, -74.0330, "Car-free walkway, best straight-on Manhattan view anywhere."),
             S("Grove St, Jersey City", 40.7195, -74.0430, "Dense cluster of coffee and food.")],
      warnings=["From a New York start, ferry and PATH schedules are the real constraint.",
                "PATH restricts bikes during weekday rush hours."],
      waypoints=[(40.7150, -74.0330), (40.7110, -74.0410), (40.7050, -74.0550),
                 (40.7020, -74.0500), (40.7080, -74.0560)],
      gmaps_dest="Liberty State Park, Jersey City, NJ"),

    R(name="Floyd Bennett Field",
      blurb="An abandoned airfield. No cars, no lights, no pedestrians.",
      base_total=32.0, base_approach=14.0, ride_miles="Runways plus a ~1.4 mi loop road",
      elevation_ft=60, surface="Cracked runway concrete plus smooth loop road",
      interruptions="Essentially none. That is the entire point.",
      interruption_score=1, car_free=True, ferry_needed=False,
      exposure="Brutally exposed — flat coastal plain, nothing blocks it",
      best_for=["Intervals", "Time trial", "Threshold testing", "Aero practice"],
      difficulty="Moderate", typical_zone="Z4 · Threshold",
      stats=[("Opened", "1931 — New York City's first municipal airport"),
             ("Site size", "~1,300 acres, inside Gateway National Recreation Area"),
             ("Straight tarmac", "Original runways up to roughly a mile"),
             ("Traffic signals on site", "Zero"),
             ("Flew from here", "Wrong Way Corrigan and Howard Hughes"),
             ("Nearest shop", "None. Carry everything.")],
      approach={"ftgreene": "Ocean Parkway greenway south, Avenue U east, Flatbush Ave south — 14 mi mostly protected.",
                "barclays": "Long. Manhattan Bridge then the full Brooklyn transit — realistically a 60+ mile day.",
                "exchange": "Impractical by bike. PATH plus subway with the bike, or pick Liberty State Park instead."},
      on_route=["Entrance is on **Flatbush Ave**, on the left just before the Marine Parkway "
                "Bridge approach.",
                "**Loop road** for smooth sustained efforts; **old runways** for all-out.",
                "Run efforts **out and back** in both directions — otherwise your numbers are "
                "a wind reading, not a fitness reading."],
      description=(
          "The most useful venue in this app, and almost nobody outside the local racing "
          "scene trains here.\n\n"
          "Floyd Bennett was New York's first municipal airport, opened 1931, now a largely "
          "abandoned 1,300-acre expanse inside Gateway National Recreation Area. Old runways "
          "give you close to a mile of dead-straight, dead-flat tarmac with no cars, no "
          "signals, no pedestrians, and frequently nobody at all.\n\n"
          "**This is where you get a real threshold heart rate.** Every other venue "
          "contaminates the test: park loops make you brake, streets make you stop, hills "
          "swing HR 30 bpm either side of your actual effort. Here nothing interrupts a "
          "20-minute effort, which is exactly what the test requires.\n\n"
          "The catch is wind. It is a flat coastal plain with nothing upwind, so check the "
          "forecast before you ride 14 miles to test yourself in a 20 mph blow."),
      tips=["**The threshold test:** warm up 15 min, then 20 min as hard as you can hold "
            "*steadily*. Average HR over the final 15 of those 20 approximates your LTHR. "
            "Enter it in the sidebar.",
            "**Do the test on a calm day.** Under 8 mph wind or the number is fiction. "
            "The forecast tab will find you a window.",
            "Loop road for sustained work; the runway concrete is cracked and seamed.",
            "Sand drifts across the runways after storms. Scan ahead.",
            "28mm tires minimum. Not a place for 23mm race rubber.",
            "**Pair it with Rockaway** — it sits at the foot of the Marine Parkway Bridge.",
            "No shop, no bodega, limited water. Two tubes and a pump."],
      stops=[S("Floyd Bennett runways", 40.5910, -73.8930, "The old tarmac. Your interval venue."),
             S("Hangar B", 40.5890, -73.8890, "Volunteer-restored vintage aircraft. Free, limited hours."),
             S("Aviator Sports", 40.5880, -73.8990, "The only reliable bathroom, water and vending on site."),
             S("Marine Park Salt Marsh", 40.5990, -73.9280, "On the approach. Bathrooms, water, trails.")],
      warnings=["Genuinely remote for New York. Carry two tubes, a pump, and the ability to fix "
                "a flat yourself.",
                "Do not run the LTHR test at the end of a 14-mile approach. Spin out easy, test fresh."],
      waypoints=[(40.5990, -73.9700), (40.5990, -73.9280), (40.5920, -73.9020),
                 (40.5895, -73.8888), (40.5910, -73.8930)],
      gmaps_dest="Floyd Bennett Field, Brooklyn, NY",
      gmaps_via=["Ocean Parkway Bike Path, Brooklyn, NY"]),

    R(name="Shore Parkway Greenway",
      blurb="Best pavement in Brooklyn. Zero cars, harbor on your right.",
      base_total=30.0, base_approach=8.5, ride_miles="~7 mi each way along the Narrows",
      elevation_ft=150, surface="Smooth separated greenway",
      interruptions="Low once you are on the path",
      interruption_score=2, car_free=True, ferry_needed=False,
      exposure="Exposed — open water on one side",
      best_for=["Endurance", "Tempo", "Long ride", "Steady-state"],
      difficulty="Easy-Moderate", typical_zone="Z3 · Tempo",
      stats=[("Greenway", "~7 mi, Owl's Head south and east toward Bath Beach"),
             ("Elevation", "~150 ft round trip — effectively flat"),
             ("Verrazzano-Narrows Bridge", "693 ft towers, 4,260 ft main span. You ride under it."),
             ("Surface", "Rated the best pavement in Brooklyn by local riders"),
             ("Connects to", "The Rockaway route, eastbound")],
      approach={"ftgreene": "4th Ave protected lane south through Park Slope and Sunset Park, 8.5 mi.",
                "barclays": "Manhattan Bridge then 4th Ave south — about 12 mi, a real approach.",
                "exchange": "Awkward. PATH to WTC, then the full length of Brooklyn. Liberty State Park is the better call."},
      on_route=["Enter at **Owl's Head Park** or the **69th St Pier**.",
                "Greenway runs **south under the Verrazzano**, then east toward Bath Beach.",
                "Fully separated from traffic the whole way — hold a steady effort here."],
      description=(
          "The best-kept pavement in the borough. A fully separated greenway along the "
          "Narrows with the harbor on one side and the Belt Parkway on the other: smooth, "
          "uninterrupted surface, no cars.\n\n"
          "Riding directly beneath the Verrazzano-Narrows Bridge is one of those moments "
          "that makes urban cycling worth the hassle. The towers go up 693 feet.\n\n"
          "Flat, smooth and mostly free of interruptions makes this one of the few places "
          "in Brooklyn where a 20-minute steady effort produces numbers that mean "
          "something — and unlike Floyd Bennett it does not cost you a 14-mile approach."),
      tips=["**4th Ave is the approach.** Not scenic, but the protected lane is fast. Avoid "
            "3rd Ave, which is a truck route.",
            "Pedestrians thin out dramatically south of the bridge.",
            "**Best sunset ride in Brooklyn**, no contest.",
            "Owl's Head has a short steep climb if you want vertical on a flat day.",
            "Bay Ridge's 5th Ave is the best refuel strip of any south Brooklyn route.",
            "Exposed to wind off the water — a southerly makes the return leg work."],
      stops=[S("69th St Pier", 40.6320, -74.0290, "American Veterans Memorial Pier. Huge harbor views, ferry to Manhattan."),
             S("Owl's Head Park", 40.6390, -74.0330, "Short punchy climb, harbor overlook, greenway access."),
             S("Under the Verrazzano", 40.6060, -74.0400, "Directly beneath the towers. Worth stopping."),
             S("Bay Ridge 5th Ave", 40.6250, -74.0270, "Bakeries, delis, Middle Eastern food."),
             S("718 Cyclery", 40.6560, -73.9880, "461 7th Ave, South Slope. Custom builds and repair workshops.")],
      warnings=["Sections of the Belt Parkway greenway go under repair periodically. Check for "
                "closures before a long day."],
      waypoints=[(40.6390, -74.0330), (40.6320, -74.0290), (40.6180, -74.0360),
                 (40.6060, -74.0400), (40.5990, -74.0100)],
      gmaps_dest="69th Street Pier, Brooklyn, NY"),

    R(name="Rockaway Beach Round Trip",
      blurb="Forty-four miles to the Atlantic and back. The classic big day.",
      base_total=44.0, base_approach=21.0, ride_miles="Out-and-back with boardwalk options",
      elevation_ft=400, surface="Greenway, bike lane, bridge path, boardwalk",
      interruptions="One dismount zone, some street sections",
      interruption_score=3, car_free=False, ferry_needed=False,
      exposure="Exposed — ocean and bay, and an afternoon sea breeze is reliable",
      best_for=["Long ride", "Endurance", "Fueling practice", "Big day out"],
      difficulty="Hard (distance, not terrain)", typical_zone="Z2 · Endurance",
      stats=[("Round trip from Fort Greene", "~44 mi"),
             ("Komoot's version", "42.4 mi, ~394 ft climbing, ~3:06 moving"),
             ("Marine Parkway–Gil Hodges Bridge", "0.75 mi span, opened 1937"),
             ("Bridge distinction", "Longest vertical-lift bridge open to motor traffic when built"),
             ("Toll for cyclists", "Free"),
             ("Bail-out", "The A train from Rockaway carries bikes")],
      approach={"ftgreene": "Ocean Parkway greenway south to Brighton Beach, then the Belt greenway east — 21 mi to the bridge.",
                "barclays": "Adds 8 mi each way over the Manhattan Bridge. A 60-mile day.",
                "exchange": "Not realistic as a ride. Train the bike to Brooklyn first, or pick another route."},
      on_route=["**Ocean Parkway** greenway south to **Brighton Beach**.",
                "East on Emmons Ave to the **Belt Parkway Greenway**, past Plumb Beach.",
                "**Flatbush Ave** south — cross at the greenway crossing, do not merge.",
                "**Marine Parkway–Gil Hodges Bridge**, bike path on the west side.",
                "Off the bridge you are at **Jacob Riis Park**. Boardwalk riding is legal here.",
                "Cut to **Rockaway Beach Blvd** — bike lane runs east to B 116th St."],
      description=(
          "The signature Brooklyn long ride. Roughly 44 miles door to door, almost entirely "
          "flat except two bridge climbs, ending with the Atlantic Ocean and tacos.\n\n"
          "The route strings together the best infrastructure in south Brooklyn: the 1894 "
          "Ocean Parkway path, the Belt Parkway greenway along Jamaica Bay, then the Gil "
          "Hodges Bridge into Queens.\n\n"
          "**This is the ride where fueling stops being optional.** At three hours of moving "
          "time you are well past what stored glycogen covers, and the failure shows up in "
          "the last ten miles rather than when you made the mistake.\n\n"
          "It is also the ride where the sea breeze is a near-certainty. Afternoons build an "
          "onshore wind, which means a headwind for the entire return leg. Go early."),
      tips=["**Go early.** The afternoon sea breeze means a headwind home, and the bridge path "
            "and greenway both congest by midday on weekends.",
            "**Start eating at minute 30**, not when you feel empty. Hunger lags need by 30–45 min.",
            "**Do not accidentally ride onto the Belt Parkway itself.** The transitions are "
            "poorly signed. If you are on a road with 55 mph traffic, stop and turn around.",
            "The A train back is a legitimate plan, not a failure.",
            "Two tubes. You are a long way from a shop for most of this.",
            "**Cardiac drift is the trap here.** Expect HR to climb 5–10 bpm over three hours "
            "at identical effort. Chase the number down and you will finish having ridden far "
            "easier than you planned. Judge by breathing and legs late on."],
      stops=[S("Brancaccio's Food Shop", 40.6530, -73.9800, "Windsor Terrace, top of Ocean Pkwy. Iced coffee and a cheese danish — the traditional fuel stop."),
             S("Plumb Beach", 40.5800, -73.9200, "Greenway waypoint on Jamaica Bay. Horseshoe crabs in spring."),
             S("Marine Parkway Bridge", 40.5830, -73.8850, "0.75 mi vertical-lift bridge, 1937. Path on the west side, 360° views."),
             S("Jacob Riis Park", 40.5680, -73.8770, "Art Deco bathhouse, wide beach, legal boardwalk riding, bathrooms."),
             S("Fort Tilden / Battery Harris", 40.5650, -73.8930, "Decommissioned WWII coastal battery. Climb it for ocean-to-skyline views."),
             S("Rockaway Taco, B 96th", 40.5850, -73.8130, "The traditional turnaround. Fish tacos. Tacoway Beach at B 87th is the other pick.")],
      warnings=["**The Marine Parkway Bridge path is narrow and technically a walkway.** Signs "
                "ask cyclists to dismount, and with pedestrians present there are inches of "
                "clearance. Early morning or late evening is much safer, and walking it costs "
                "two minutes.",
                "The Belt greenway near Gerritsen Inlet has been under construction periodically, "
                "with a narrow climb through the work zone."],
      waypoints=[(40.6540, -73.9760), (40.5990, -73.9700), (40.5780, -73.9600),
                 (40.5840, -73.9430), (40.5800, -73.9200), (40.5895, -73.8888),
                 (40.5830, -73.8850), (40.5680, -73.8770), (40.5810, -73.8370)],
      gmaps_dest="Rockaway Beach Blvd & Beach 116th St, Queens, NY",
      gmaps_via=["Marine Parkway Bridge, Brooklyn, NY"]),

    R(name="River Road / Alpine Hill",
      blurb="Car-free road between the Hudson and the Palisades. Real climbing.",
      base_total=45.0, base_approach=17.0, ride_miles="~10 mi, repeatable",
      elevation_ft=1400, surface="Good pavement, twisty, some rough patches",
      interruptions="Very low — park road with minimal traffic",
      interruption_score=1, car_free=False, ferry_needed=False,
      exposure="Sheltered by the cliffs, but the greenway approach is exposed",
      best_for=["Hill repeats", "Threshold", "Long ride", "Climbing"],
      difficulty="Hard", typical_zone="Z3 · Tempo",
      stats=[("Henry Hudson Drive", "~11 mi of near-car-free park road along the Palisades"),
             ("Round trip elevation", "~1,400–1,800 ft depending on how much you climb"),
             ("GWB bike path", "Free, ~1 mi crossing. South path is the cyclist path."),
             ("State Line Lookout", "~530 ft, the classic summit extension"),
             ("Seasonal", "River Road gates close roughly late fall to early spring")],
      approach={"ftgreene": "Manhattan Bridge, 2nd Ave north, then the Hudson greenway to the GWB — 17 mi.",
                "barclays": "West to the greenway then straight north to the GWB — about 9 mi, and all of it car-free.",
                "exchange": "River Road is on your side of the water. Ride north through Hoboken and Fort Lee, roughly 12 mi."},
      on_route=["Cross the **George Washington Bridge** on the south path.",
                "Descend to **Henry Hudson Drive (River Road)** and head north along the river.",
                "**Alpine Hill** is the benchmark climb back out.",
                "**State Line Lookout** is the extension if you want the full summit."],
      description=(
          "The local serious-cyclist playground, and the reason New York riders do not "
          "complain more about living in a flat city.\n\n"
          "Over the George Washington Bridge, Henry Hudson Drive drops you onto a narrow "
          "twisting park road wedged between the Hudson and the Palisades cliffs. Almost no "
          "car traffic, decent surface, constant climbing and descending. Alpine Hill is the "
          "standard benchmark ascent back out.\n\n"
          "**Start point changes this one a lot.** From Barclays the approach is nine "
          "car-free greenway miles. From Exchange Place you skip the bridge entirely and "
          "ride up your own side of the river. From Fort Greene it is 17 miles each way.\n\n"
          "Forty-five miles with 1,400+ feet of climbing and long technical descents. Not "
          "your first ride on a new bike."),
      tips=["**Strictly Bicycles** at the foot of the bridge is a full shop plus café and the "
            "unofficial clubhouse for this ride.",
            "**Descents are technical** — tight switchbacks, occasional gravel, real speed. "
            "Ride the first descent slow and learn the corners.",
            "**Check River Road is open** before riding out. Gates close seasonally and after storms.",
            "Layers. It is meaningfully cooler on River Road, and you sweat climbing then "
            "freeze descending.",
            "A whole-ride HR average is meaningless here. Look at the climbs alone."],
      stops=[S("Strictly Bicycles", 40.8540, -73.9450, "2347 Hudson Terrace, Fort Lee. Shop, café, unofficial HQ."),
             S("Little Red Lighthouse", 40.8500, -73.9470, "Under the GWB on the Manhattan side."),
             S("Alpine Boat Basin", 40.9450, -73.9210, "River-level rest stop. Water and bathrooms."),
             S("State Line Lookout", 41.0130, -73.9070, "Top of the Palisades, ~530 ft. Small café."),
             S("Dinosaur BBQ", 40.8180, -73.9600, "131st & 12th Ave, off the greenway approach.")],
      warnings=["Long technical descents on a road you do not know. First time down, ride it slow.",
                "River Road closes seasonally and after storms. Verify before committing to the approach."],
      waypoints=[(40.8000, -73.9720), (40.8500, -73.9470), (40.8517, -73.9527),
                 (40.8540, -73.9450), (40.9000, -73.9250), (40.9450, -73.9210),
                 (41.0130, -73.9070)],
      gmaps_dest="Henry Hudson Drive, Alpine, NJ",
      gmaps_via=["George Washington Bridge, New York, NY"]),

    R(name="9W to Nyack",
      blurb="The region's roadie superhighway. Thirty miles north on a signed bike route.",
      base_total=95.0, base_approach=17.0, ride_miles="~66 mi round trip from the GWB",
      elevation_ft=2800, surface="Wide-shouldered highway designated a state bike route",
      interruptions="Low — 9W has a big shoulder and is built for this",
      interruption_score=2, car_free=False, ferry_needed=False,
      exposure="Mixed — wooded stretches, but exposed on the ridges",
      best_for=["Epic long ride", "Endurance", "Group riding", "Bucket list"],
      difficulty="Very hard (distance)", typical_zone="Z2 · Endurance",
      stats=[("GWB to Nyack", "~30 mi each way"),
             ("Designation", "NY State Bike Route 9 — continues 345 mi to Montreal"),
             ("Lollipop variant", "44.4 mi / +2,485 ft from the GWB via Bradley-Tweed and Blauvelt"),
             ("Full menu", "73 mi / ~5,800 ft via Hook Mountain and Rockland Lake"),
             ("Trains take bikes", "NJ Transit and Metro-North both work as exits")],
      approach={"ftgreene": "17 mi to the GWB via the Manhattan Bridge and Hudson greenway. A 95-mile day.",
                "barclays": "9 mi of car-free greenway to the GWB. Turns this into a 78-mile day — the best start for it.",
                "exchange": "12 mi north through Hoboken and Fort Lee, no bridge crossing needed. About 80 miles total."},
      on_route=["Over the **GWB**, follow signs for **NY Bike Route 9** on Hudson Terrace.",
                "Left on **E Palisade Ave**, right onto **Sylvan Way / 9W**.",
                "**9W north ~30 mi to Nyack.** It is signed and obvious.",
                "**Coming home take Piermont Road**, not 9W — prettier, quieter, and 9W narrows "
                "unpleasantly near town."],
      description=(
          "The ride every New York cyclist eventually does. 9W is a wide-shouldered highway "
          "that happens to be New York State Bike Route 9 — the same route that runs 345 "
          "miles to Montreal — and on a weekend morning it carries so many cyclists it "
          "functions as an informal peloton.\n\n"
          "Rolling rather than mountainous: roughly 2,500 feet over the 66-mile round trip "
          "from the GWB, spread across long gradual grades.\n\n"
          "**Your start point decides whether this is sane.** From Barclays it is 78 miles "
          "with a nine-mile car-free approach. From Fort Greene it is 95 miles with 17 miles "
          "of city on each end. Shorten the turnaround to Piermont, or ride out and take the "
          "train home — both entirely normal.\n\n"
          "The Rapha clubhouse group rides head out this way, if you want company for a "
          "first attempt."),
      tips=["**Build to it.** Several 40–50 mile rides first. The failure mode is not fitness, "
            "it is being 60 miles from home with nothing left.",
            "**The Runcible Spoon** in Nyack is the traditional turnaround. Closes 6pm.",
            "Piermont is the better turnaround for a shorter day, and **Bunbury's** there has "
            "been the cyclist stop for decades.",
            "**Fuel every 25–30 minutes from the start.** Five hours means you hit the "
            "absorption ceiling — read the fueling tab before you pack.",
            "Shops: Strictly Bicycles at the start, then very little until Nyack.",
            "**Ride the first two hours easier than feels right.** Restraint early is the whole "
            "discipline, and HR is the honest check when fresh legs are lying to you."],
      stops=[S("Strictly Bicycles", 40.8540, -73.9450, "Fort Lee, foot of the GWB. Last real shop."),
             S("Bunbury's Coffee Shop", 41.0410, -73.9190, "460 Piermont Ave. The decades-old cyclist stop."),
             S("Coffee Ride Café", 41.0800, -73.9200, "South Nyack, the former village hall. Founded by cyclists, big bike parking."),
             S("The Runcible Spoon", 41.0900, -73.9170, "Nyack. The classic turnaround bakery. Closes 6pm."),
             S("Piermont Pier", 41.0400, -73.9080, "Mile-long pier into the Hudson.")],
      warnings=["A full day with real consequences if you bonk far from home. Carry more food "
                "than you think and have a train plan.",
                "Hudson crossings for bikes are limited. Do not improvise a route across the river."],
      waypoints=[(40.8517, -73.9527), (40.8540, -73.9450), (40.9200, -73.9500),
                 (41.0000, -73.9350), (41.0410, -73.9190), (41.0908, -73.9179)],
      gmaps_dest="Nyack, NY",
      gmaps_via=["George Washington Bridge, New York, NY"]),

    R(name="Jamaica Bay + Shirley Chisholm",
      blurb="Landfill hills with harbor views. Nearly empty on weekdays.",
      base_total=34.0, base_approach=10.0, ride_miles="~12 mi of greenway and park trails",
      elevation_ft=250, surface="Paved greenway plus some hard-packed trail",
      interruptions="Low once on the greenway",
      interruption_score=2, car_free=True, ferry_needed=False,
      exposure="Exposed — open bay, no shelter on the mounds",
      best_for=["Long ride", "Endurance", "Hill repeats", "Solitude"],
      difficulty="Moderate", typical_zone="Z2 · Endurance",
      stats=[("Park size", "407 acres"),
             ("Hours", "9:00 am to dusk — gates close outside that"),
             ("Built on", "The capped Pennsylvania and Fountain Avenue landfills"),
             ("Trails", "~10 mi within the park"),
             ("Opened", "2019, still relatively unknown"),
             ("Full greenway loop", "~45 mi with ~525 ft climbing")],
      approach={"ftgreene": "Atlantic Ave east, then south to the Belt greenway — 10 mi, and East New York is the dull part.",
                "barclays": "Manhattan Bridge then across Brooklyn. About 18 mi each way.",
                "exchange": "Far. Consider Liberty State Park instead."},
      on_route=["Pick up the **Belt Parkway Greenway** heading east along Jamaica Bay.",
                "**Shirley Chisholm State Park** entrance at Pennsylvania Ave.",
                "Two capped mounds give you the climbs. **Fountain Ave overlook** is the high point.",
                "4–6 repeats of the main mound is a legitimate hill session."],
      description=(
          "The most underrated ride within easy reach of Brooklyn.\n\n"
          "Shirley Chisholm State Park is 407 acres built on the capped Pennsylvania and "
          "Fountain Avenue landfills — which sounds grim and is the whole reason it works. "
          "Capping two enormous mounds of garbage produced something Brooklyn otherwise does "
          "not have: sustained rolling climbs with unobstructed views over Jamaica Bay and "
          "the Manhattan skyline.\n\n"
          "Opened in 2019 and still relatively unknown, so on a weekday evening you can have "
          "most of it to yourself. The birdlife is extraordinary — Jamaica Bay is one of the "
          "most important migratory stops on the Atlantic flyway."),
      tips=["**Check the hours.** 9am to dusk, gates locked outside that. Do not ride ten miles "
            "to find a closed gate.",
            "Short repeatable climbs make this a real hill session without leaving Brooklyn.",
            "Atlantic to Van Sinderen is calmer than Rockaway Ave on the approach.",
            "**Weekday evenings are near-empty.** Weekend mornings bring families.",
            "Water fountains but no food. Bring your own.",
            "The mounds are fully exposed — a windy day makes the climbs meaningfully harder."],
      stops=[S("Shirley Chisholm State Park", 40.6440, -73.8830, "407 acres, 10 mi of trails, real climbs, skyline views. 9am–dusk."),
             S("Fountain Ave overlook", 40.6410, -73.8720, "Top of the eastern mound. Best view in the park."),
             S("Canarsie Pier", 40.6280, -73.8850, "Gateway NRA pier on Jamaica Bay. Bathrooms, water, seasonal food."),
             S("Belt Pkwy Greenway", 40.6350, -73.8900, "Flat separated path. Connects west to Bay Ridge, east to Rockaway.")],
      warnings=["Some park trails are hard-packed gravel. Fine on 28mm+, sketchy on narrow tires."],
      waypoints=[(40.6500, -73.8950), (40.6350, -73.8900), (40.6440, -73.8830),
                 (40.6410, -73.8720), (40.6280, -73.8850)],
      gmaps_dest="Shirley Chisholm State Park, Brooklyn, NY"),

    R(name="Brooklyn Waterfront Greenway",
      blurb="The short shakeout. Skyline views, cobblestone hazard.",
      base_total=14.0, base_approach=1.5, ride_miles="~8 mi of waterfront",
      elevation_ft=120, surface="Mixed greenway, protected lane, DUMBO cobbles",
      interruptions="Pedestrian-heavy through Brooklyn Bridge Park",
      interruption_score=4, car_free=False, ferry_needed=False,
      exposure="Partly exposed along the water",
      best_for=["Recovery spin", "Bike handling", "New-bike shakedown", "Weeknight ride"],
      difficulty="Easy", typical_zone="Z1 · Recovery",
      stats=[("Greenway", "~8 mi, Greenpoint down to Sunset Park"),
             ("Best segment", "Brooklyn Bridge Park waterfront into Columbia St"),
             ("Elevation", "~120 ft — short ramps, nothing sustained"),
             ("What it is good for", "Shaking down a new bike close to home")],
      approach={"ftgreene": "Straight down through the Navy Yard to the water, 1.5 mi.",
                "barclays": "Manhattan Bridge and you are in DUMBO — about 5 mi.",
                "exchange": "PATH to WTC then the Brooklyn Bridge. Not the natural fit."},
      on_route=["Pick up the waterfront path through **Brooklyn Bridge Park**.",
                "South on the **Columbia St** protected lane toward Red Hook.",
                "**Louis Valentino Pier** is the quiet payoff at the end."],
      description=(
          "Not a training venue so much as a utility ride — the 'I have 45 minutes' option. "
          "More importantly it is where you should do the first rides on any new bike.\n\n"
          "You get cobbles, tight turns, pedestrian chaos and a couple of punchy ramps "
          "compressed into a mile and a half from home. If your saddle height is wrong or "
          "your bars are too low, you will know inside twenty minutes and you will be close "
          "enough to bail. Better place to find a fit problem than 22 miles out on the Belt.\n\n"
          "Views are absurd — directly beneath both bridges with Lower Manhattan across the "
          "water."),
      tips=["**DUMBO's cobblestones will rattle your teeth out.** Walking pace on Washington "
            "and Water St, or route around via Front St.",
            "Brooklyn Bridge Park is pedestrian-priority and packed on nice evenings.",
            "Red Hook is the quiet reward — empty industrial streets on weekend mornings.",
            "Good route for testing lights before riding anywhere serious after dark.",
            "If your HR will not stay down here, that is the route's fault, not your "
            "discipline. Stoplights and cobbles spike it regardless."],
      stops=[S("Redbeard Bikes", 40.7030, -73.9860, "18 Bridge St, DUMBO. Known for careful bike fitting."),
             S("Brooklyn Roasting Company", 40.7030, -73.9870, "25 Jay St. Bike parking, Dough donuts."),
             S("Louis Valentino Jr Pier", 40.6810, -74.0130, "Red Hook. Straight-on Statue of Liberty view, almost always empty."),
             S("Pier 6, Brooklyn Bridge Park", 40.6930, -74.0000, "Ferry dock, water, bathrooms.")],
      warnings=["Cobblestones plus narrow tires plus pedestrians is how people go down."],
      waypoints=[(40.7030, -73.9870), (40.6930, -74.0000), (40.6850, -74.0060),
                 (40.6810, -74.0130)],
      gmaps_dest="Louis Valentino Jr. Park and Pier, Brooklyn, NY"),

    R(name="Coney Island via Ocean Parkway",
      blurb="America's first bike path, straight to the boardwalk. Dead flat.",
      base_total=24.0, base_approach=12.0, ride_miles="Out-and-back, ~5 mi of path",
      elevation_ft=90, surface="Ocean Parkway greenway, some root heave",
      interruptions="Cross-street signals the length of Ocean Pkwy",
      interruption_score=4, car_free=False, ferry_needed=False,
      exposure="Sheltered on the parkway, exposed at the beach",
      best_for=["Endurance", "Long ride", "Beginner-friendly"],
      difficulty="Easy", typical_zone="Z2 · Endurance",
      stats=[("Path opened", "1894 — the first dedicated bike path in the United States"),
             ("Designed by", "Olmsted and Vaux, same pair as Prospect Park"),
             ("Path length", "~5 mi, Church Ave to Surf Ave"),
             ("Elevation", "Under 100 ft round trip"),
             ("Character", "Impossible to get lost. One straight line south.")],
      approach={"ftgreene": "Through Prospect Park to Park Circle, then the path starts.",
                "barclays": "Manhattan Bridge then Flatbush and Prospect Park. Around 18 mi each way.",
                "exchange": "Long transit. Better options on your side."},
      on_route=["Pick up the **Ocean Parkway bike path** at **Park Circle**, west side of the parkway.",
                "**Straight south ~5 mi.** The path ends near Surf Ave.",
                "**Coney Island boardwalk**, or turn east to Brighton Beach and Sheepshead Bay."],
      description=(
          "The simplest long-ish ride in Brooklyn. One straight line south for five miles on "
          "a path that has existed since 1894, built as the first dedicated bicycle path in "
          "the United States, designed by Olmsted and Vaux.\n\n"
          "Dead flat, tree-shaded, and the payoff is the boardwalk with the Atlantic in front "
          "of you.\n\n"
          "The tradeoff is signals. Ocean Parkway crosses a numbered street every couple of "
          "blocks, so your average speed reads low and it is not a reflection of your fitness."),
      tips=["**Do not chase average speed here.** The lights make it structurally impossible. "
            "This is a duration ride.",
            "Root heave and pavement seams — fine on 28mm, jarring on 23mm.",
            "**Brighton Beach is the better food stop** than Coney Island. Russian bakeries and "
            "grocers, cheap and carb-dense.",
            "Afternoon sea breeze means a headwind home. Ride out early.",
            "Boardwalk cycling is time-restricted, generally early morning. Check posted signs.",
            "Watch HR rather than speed — the lights wreck pace but HR still reports the riding honestly."],
      stops=[S("Park Circle", 40.6540, -73.9760, "Southwest corner of Prospect Park. The path starts here."),
             S("Coney Island Boardwalk", 40.5730, -73.9800, "Riegelmann Boardwalk, Nathan's, the Cyclone."),
             S("Brighton Beach Ave", 40.5780, -73.9600, "Russian bakeries and grocers. Cheap, carb-dense refueling."),
             S("Sheepshead Bay", 40.5840, -73.9430, "Fishing boats and seafood. Links east toward Rockaway.")],
      warnings=["Boardwalk cycling is time-restricted. Do not assume it is allowed when you arrive."],
      waypoints=[(40.6540, -73.9760), (40.6250, -73.9720), (40.5990, -73.9700),
                 (40.5780, -73.9600), (40.5730, -73.9800)],
      gmaps_dest="Coney Island Boardwalk, Brooklyn, NY",
      gmaps_via=["Ocean Parkway Bike Path, Brooklyn, NY"]),

    R(name="Brooklyn Café & Shop Crawl",
      blurb="A social ride disguised as training. Coffee, shops, the local scene.",
      base_total=22.0, base_approach=0.0, ride_miles="~22 mi loop", is_loop=True,
      elevation_ft=380, surface="Protected lanes, greenway, rough Bushwick street",
      interruptions="Constant. This is a city ride with real stoplights.",
      interruption_score=5, car_free=False, ferry_needed=False,
      exposure="Sheltered by buildings throughout",
      best_for=["Social ride", "Recovery spin", "Weeknight ride", "Sightseeing"],
      difficulty="Easy", typical_zone="Z1 · Recovery",
      stats=[("Loop", "~22 mi with all stops"),
             ("Route", "DUMBO → Williamsburg → Bushwick → Bed-Stuy → back"),
             ("Elevation", "~380 ft, mostly the Williamsburg Bridge"),
             ("Elapsed vs moving", "3–4 hours out, ~1:30 of actual riding"),
             ("Shops on route", "Redbeard, Bicycle Roots, Fulton Bikes")],
      approach={"ftgreene": "Starts at the door. This is the Brooklyn local's loop.",
                "barclays": "Manhattan Bridge in, then run the loop. Adds ~10 mi.",
                "exchange": "PATH to WTC, Manhattan Bridge, then the loop."},
      on_route=["West to **DUMBO** — Redbeard Bikes and Brooklyn Roasting.",
                "North along the waterfront to **Williamsburg** — Domino Park and Devoción.",
                "East on **Grand St** into **Bushwick** — Sey Coffee.",
                "South through **Bed-Stuy** — Bicycle Roots on Franklin, Fulton Bikes on Fulton.",
                "West on Fulton / Lafayette back to Fort Greene."],
      description=(
          "Every training plan needs a ride that is not training. This is that ride.\n\n"
          "A 22-mile loop stringing together Brooklyn's cycling and coffee culture: DUMBO's "
          "shops and roasters, the Williamsburg waterfront, Bushwick's third-wave coffee, the "
          "Bed-Stuy corridor. You will spend more time off the bike than on it.\n\n"
          "It is also reconnaissance. You meet the shops you will eventually need — a fit "
          "adjustment, a wheel true, an emergency tube on a Sunday — and you learn which "
          "Brooklyn bike lanes are actually pleasant versus which merely exist on a map.\n\n"
          "Also the correct choice on a genuinely bad weather day. Buildings shelter you, and "
          "nobody cares what your average speed was."),
      tips=["**Bring a lock.** Every other route here you never leave the bike; this one you "
            "leave it constantly.",
            "Weekday mornings if you actually want to talk to shop staff.",
            "**Devoción** roasts fresh-crop Colombian in a glass-roofed room. Best coffee on the loop.",
            "**Sey Coffee** in Bushwick is the other serious one, reliably full of cyclists.",
            "Bushwick streets are rough — cobbles, potholes, freight rail crossings.",
            "**Ignore HR entirely.** Stop-start city riding produces a garbage average."],
      stops=[S("Redbeard Bikes", 40.7030, -73.9860, "18 Bridge St, DUMBO. Bike fitting and after-sale support."),
             S("Brooklyn Roasting Company", 40.7030, -73.9870, "25 Jay St. Bike parking, Dough donuts."),
             S("Domino Park", 40.7145, -73.9680, "Williamsburg waterfront. Skyline through the old sugar refinery."),
             S("Devoción", 40.7150, -73.9620, "69 Grand St. Glass-roofed room, fresh-crop Colombian."),
             S("Sey Coffee", 40.7070, -73.9330, "18 Grattan St, Bushwick. Light roasts, full of cyclists."),
             S("Bicycle Roots", 40.6790, -73.9560, "663 Franklin Ave, Crown Heights. Community shop."),
             S("Fulton Bikes", 40.6830, -73.9600, "997 Fulton St, Bed-Stuy. On the way home.")],
      warnings=["Highest traffic exposure here. Ride defensively crossing Flushing Ave and along "
                "Grand St.",
                "Bike theft is real in these neighborhoods. Never leave a bike unlocked, even briefly."],
      waypoints=[(40.7030, -73.9865), (40.7080, -73.9700), (40.7145, -73.9680),
                 (40.7150, -73.9620), (40.7070, -73.9330), (40.6900, -73.9400),
                 (40.6790, -73.9560), (40.6830, -73.9600)],
      gmaps_dest="Devocion, 69 Grand St, Brooklyn, NY",
      gmaps_via=["Redbeard Bikes, 18 Bridge St, Brooklyn, NY", "Domino Park, Brooklyn, NY"]),

    R(name="Governors Island",
      blurb="2.2 car-free miles in the middle of the harbor.",
      base_total=10.0, base_approach=2.5, ride_miles="2.2 mi car-free loop", 
      elevation_ft=40, surface="Smooth, fully car-free",
      interruptions="Pedestrian-heavy on weekends",
      interruption_score=3, car_free=True, ferry_needed=True,
      exposure="Fully exposed — an island in open harbor",
      best_for=["Recovery spin", "Social ride", "Sightseeing"],
      difficulty="Easy", typical_zone="Z1 · Recovery",
      stats=[("Island", "172 acres"),
             ("Loop", "2.2 mi, entirely car-free"),
             ("Ferry from", "Brooklyn Bridge Park Pier 6, or the Battery Maritime Building"),
             ("Season", "Open year-round, hours vary seasonally"),
             ("Bikes on ferry", "Permitted, small fee"),
             ("The Hills", "Constructed landform at the south end. Small climbs, big panorama.")],
      approach={"ftgreene": "2.5 mi to Pier 6, then the ferry.",
                "barclays": "Downtown to the Battery Maritime Building, about 6 mi.",
                "exchange": "PATH to WTC then east to the Battery. Doable but fiddly."},
      on_route=["Ferry to **Governors Island** — bikes allowed, small fee.",
                "Ride the **2.2 mi perimeter loop counter-clockwise** for the best skyline sequence.",
                "**The Hills** at the south end are the one bit of climbing."],
      description=(
          "Not training. A nice thing to do on a bike.\n\n"
          "172 acres in the middle of New York Harbor, ten minutes by ferry, and no cars at "
          "all. The perimeter loop gives you Lower Manhattan, the Statue of Liberty, the "
          "Verrazzano and Brooklyn in sequence — probably the best set of views available "
          "from a bicycle anywhere in the city.\n\n"
          "Use it for an easy spin, or when someone is visiting. Do not plan a workout around "
          "a 2.2-mile loop shared with families on rented tandems."),
      tips=["**Go on a weekday** if you want to ride rather than weave.",
            "Ferry schedules are limited and the last boat back is early. Check before you go.",
            "Free ferry hours exist some mornings. Worth checking.",
            "The Hills have the best harbor panorama on the island.",
            "Food is seasonal and expensive. Bring your own.",
            "Castle Williams (1811) is free and takes ten minutes."],
      stops=[S("Pier 6, Brooklyn Bridge Park", 40.6930, -74.0000, "Ferry departure. Bathrooms, water."),
             S("The Hills", 40.6860, -74.0200, "South end. Small climbs, huge views."),
             S("Castle Williams", 40.6910, -74.0180, "1811 circular fortification. Free to enter."),
             S("Colonels Row", 40.6900, -74.0160, "Shaded lawns and old officers' housing.")],
      warnings=["Seasonal and limited ferry hours. Confirm the last departure before you go."],
      waypoints=[(40.6930, -74.0000), (40.6895, -74.0170), (40.6860, -74.0200),
                 (40.6910, -74.0180)],
      gmaps_dest="Brooklyn Bridge Park Pier 6, Brooklyn, NY"),
]


# ============================================================================
# SIDEBAR
# ============================================================================

st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h3 style='margin-bottom:0'>◎ Ride Finder</h3>", unsafe_allow_html=True)
    eyebrow("New York · road cycling")

    st.divider()
    eyebrow("How much detail?")
    MODE = st.radio(
        "Detail level", ["Just riding", "Training", "Full data"],
        label_visibility="collapsed",
        help="Start simple. Turn up the detail when you want it — nothing is lost, "
             "it is only hidden.",
    )
    st.caption({
        "Just riding": "Where to go today, how long it takes, and whether you will get wet.",
        "Training": "Adds heart rate zones and fueling for the ride you pick.",
        "Full data": "Everything, including the equations and their error bars.",
    }[MODE])

    st.divider()
    eyebrow("Start point")
    start_key = st.radio(
        "Start point", list(STARTS.keys()),
        format_func=lambda k: STARTS[k].label,
        label_visibility="collapsed",
    )
    SP = STARTS[start_key]
    st.caption(SP.note)

    st.divider()
    eyebrow("Rider")
    if MODE == "Just riding":
        pace_word = st.select_slider(
            "How fast do you usually ride?",
            options=["Gentle", "Steady", "Brisk", "Fast"], value="Steady",
            help="Rough guide: Gentle is a relaxed bike-path pace, Fast is a club rider.",
        )
        avg_mph = {"Gentle": 9.5, "Steady": 12.0, "Brisk": 15.0, "Fast": 18.0}[pace_word]
        st.caption(f"About {avg_mph:.0f} mph — used to estimate how long rides take.")
        age = int(_setting("AGE", 30))
        weight_lb = float(_setting("WEIGHT_LB", 160.0))
        sex = "Male"
    else:
        age = st.number_input("Age", 14, 90, int(_setting("AGE", 30)))
        weight_lb = st.number_input("Weight (lb)", 80.0, 400.0,
                                    float(_setting("WEIGHT_LB", 160.0)), step=1.0)
        sex = st.radio("Sex (for the energy equation)", ["Male", "Female"], horizontal=True)
        avg_mph = st.slider("Average moving speed (mph)", 9.0, 24.0, 15.0, step=0.5,
                            help="Drives time and fueling estimates. Include the slow city sections.")
    weight_kg = weight_lb * 0.45359237

    st.divider()
    if MODE == "Just riding":
        max_hr = int(_setting("MAX_HR", 190))
        rest_hr = int(_setting("REST_HR", 60))
        zone_source = "Heart rate reserve"
        lthr = 168
        edges = hrr_zone_edges(max_hr, rest_hr)
        vo2max = None
        mixed_carb = False
        st.caption("Heart rate and fueling tools are hidden at this detail level. "
                   "Switch to Training above to turn them on.")
    else:
        eyebrow("Heart rate")
        max_hr = st.number_input("Max HR (bpm)", 120, 220, int(_setting("MAX_HR", 190)),
                                 help="A measured or watch-reported value beats any age formula.")
        rest_hr = st.number_input("Resting HR (bpm)", 30, 100, int(_setting("REST_HR", 60)))
        zone_source = st.radio(
            "Zone method",
            ["Heart rate reserve", "% of max HR", "% of LTHR"],
            help="Heart rate reserve is what Apple Watch uses on Automatic. It accounts for "
                 "your resting HR, so it is the better default.",
        )
        lthr = 168
        if zone_source == "% of LTHR":
            lthr = st.number_input("LTHR (bpm)", 100, 220, 168,
                                   help="20-min all-out test: average HR over the final 15 minutes.")

        auto_edges = hrr_zone_edges(max_hr, rest_hr)
        if zone_source == "Heart rate reserve":
            with st.expander("Match my watch exactly"):
                st.caption(f"Computed from your numbers: {auto_edges}. If your watch shows "
                           "different edges, type them here.")
                e1 = st.number_input("Z1 tops at", 80, 220, auto_edges[0])
                e2 = st.number_input("Z2 tops at", 80, 220, auto_edges[1])
                e3 = st.number_input("Z3 tops at", 80, 220, auto_edges[2])
                e4 = st.number_input("Z4 tops at", 80, 220, auto_edges[3])
            edges = [e1, e2, e3, e4]
        else:
            edges = auto_edges

        vo2_known = st.checkbox("I know my VO2max / Cardio Fitness",
                                help="Adds a second independent energy estimate to cross-check.")
        vo2max = st.number_input("VO2max (ml/kg/min)", 20.0, 85.0, 48.0, step=0.5) if vo2_known else None
        mixed_carb = st.checkbox("Mixed glucose + fructose fuel",
                                 help="Raises the absorption ceiling from ~60 to ~90 g/hr. Needs gut training.")

    st.divider()
    eyebrow("Narrow it down")
    max_total = st.slider("Longest ride you want (miles)", 5, 110,
                          30 if MODE == "Just riding" else 110, step=5)
    if MODE == "Just riding":
        easy_only = st.checkbox("Beginner-friendly only", value=False,
                                help="Shorter, flatter, no ferries, nothing gnarly.")
        car_free_only = st.checkbox("Keep me away from cars", value=False)
        flat_only = st.checkbox("Flat routes only", value=False)
        treat_only = st.checkbox("Must have a coffee or food stop", value=False)
        purpose, no_ferry, low_interrupt = [], False, False
    else:
        easy_only = flat_only = treat_only = False
        purpose = st.multiselect("Riding for", sorted({p for r in ROUTES for p in r.best_for}))
        car_free_only = st.checkbox("Car-free riding only")
        no_ferry = st.checkbox("No ferry required")
        low_interrupt = st.checkbox("Open road only (for hard efforts)")

ZONES = build_zones(zone_source, max_hr, rest_hr, lthr, edges)
HRR = max(1, max_hr - rest_hr)
ZONE_LABEL = {"Heart rate reserve": "heart rate reserve",
              "% of max HR": "percent of max HR",
              "% of LTHR": "percent of LTHR"}[zone_source]


def keep(r: Route) -> bool:
    if r.total_from(SP) > max_total:
        return False
    if purpose and not set(purpose) & set(r.best_for):
        return False
    if car_free_only and not r.car_free:
        return False
    if no_ferry and r.ferry_needed:
        return False
    if low_interrupt and r.interruption_score > 2:
        return False
    if easy_only and not beginner_ok(r):
        return False
    if flat_only and flatness(r) != "Flat":
        return False
    if treat_only and not has_treat(r):
        return False
    return True


VISIBLE = [r for r in ROUTES if keep(r)]

# ============================================================================
# WEATHER DATA
# ============================================================================

WX = fetch_forecast(SP.lat, SP.lon, days=7)
HOURLY = WX["hourly"]
if WX["ok"] and HOURLY is not None and not HOURLY.empty:
    HOURLY = HOURLY.copy()
    HOURLY["score"] = HOURLY.apply(ride_score, axis=1)
    NOW = HOURLY.iloc[(HOURLY["time"] - pd.Timestamp.now()).abs().argsort()[:1]].iloc[0]
else:
    NOW = None


# ============================================================================
# HEADER
# ============================================================================

st.markdown("<h1>NYC Ride Finder</h1>", unsafe_allow_html=True)
if NOW is not None:
    st.markdown(
        "<p class='note' style='margin-top:0.35rem'>"
        "Starting from <b>{}</b> · {} of {} routes · right now {:.0f}°F, "
        "{:.0f} mph from {}, {}</p>".format(
            SP.label, len(VISIBLE), len(ROUTES), NOW.temp, NOW.wind,
            compass_point(NOW.wind_dir), score_label(NOW.score).lower()),
        unsafe_allow_html=True)
else:
    st.markdown(
        "<p class='note' style='margin-top:0.35rem'>Starting from <b>{}</b> · {} of {} "
        "routes · live weather unavailable</p>".format(SP.label, len(VISIBLE), len(ROUTES)),
        unsafe_allow_html=True)

if not VISIBLE:
    st.warning("No routes match those filters. Loosen them in the sidebar.")
    st.stop()

SHOW_FUEL = MODE != "Just riding"
_labels = ["Today & 7-day forecast", "Routes"] + (
    ["Zones & fueling calculator"] if SHOW_FUEL else [])
_tabs = st.tabs(_labels)
# Today and the forecast share one tab: today's read at the top, the week below.
TAB_TODAY = _tabs[0]
TAB_WX = _tabs[0]
TAB_ROUTE = _tabs[1]
TAB_FUEL = _tabs[2] if SHOW_FUEL else None


# ============================================================================
# TAB — TODAY
# ============================================================================

with TAB_TODAY:
    if MODE == "Just riding" and NOW is not None:
        # One clear answer first. Detail is available below for anyone who wants it.
        def _casual_rank(rt):
            hd = rt.heading_from(SP)
            h, c = wind_components(float(NOW.wind_dir), float(NOW.wind), hd)
            shelter = 6 if "Sheltered" in rt.exposure else (-8 if "Fully exposed" in rt.exposure
                                                            or "Brutally" in rt.exposure else 0)
            wind_pen = (abs(h) + c * 0.5) * (0.15 if "Sheltered" in rt.exposure else 0.9)
            comfort = 8 if rt.car_free else 0
            return NOW.score + shelter + comfort - wind_pen

        pick = max(VISIBLE, key=_casual_rank)
        ptime = pick.total_from(SP) / avg_mph
        eyebrow("Ride this today")
        st.markdown(
            f"<h2 style='margin:0.1rem 0 0.35rem'>{pick.name}</h2>"
            f"<p class='lede'>{pick.blurb}</p>", unsafe_allow_html=True)

        datastrip([
            ("Distance", f"{pick.total_from(SP):.0f} mi"),
            ("At your pace", f"{int(ptime)}h {int(round((ptime % 1) * 60)):02d}m"),
            ("Terrain", flatness(pick)),
            ("Coffee stop", "Yes" if has_treat(pick) else "No"),
        ])

        st.markdown(
            "" + rows([
                ("Weather right now", "{:.0f}°F — {}".format(NOW.temp, temp_in_words(NOW.temp))),
                ("Wind", "{:.0f} mph from {} — {}".format(
                    NOW.wind, compass_point(NOW.wind_dir), wind_in_words(NOW.wind))),
                ("Rain", rain_in_words(NOW.precip_prob, NOW.precip)),
                ("Traffic on this route", traffic_comfort(pick)),
                ("Getting there", pick.approach.get(start_key, "")),
            ]) + "</div>", unsafe_allow_html=True)

        st.link_button(f"Directions from {SP.label} ↗", pick.gmaps_url(SP))
        st.caption("Chosen for today's weather, how sheltered each route is, and how much "
                   "traffic you would share it with. Full details are on the Routes tab.")
        st.divider()

    if NOW is None:
        st.info("Live weather is unavailable, so route scoring is off. The Route detail tab "
                "still has everything else.")
    else:
        left, right = st.columns([1, 1.55])

        with left:
            eyebrow("Wind against your route")
            default_route = max(VISIBLE, key=lambda r: r.interruption_score * -1)
            rose_route = st.selectbox(
                "Route for the wind rose", [r.name for r in VISIBLE],
                index=0, label_visibility="collapsed",
            )
            rr = next(r for r in VISIBLE if r.name == rose_route)
            hd = rr.heading_from(SP)
            st.markdown(
                wind_rose_svg(hd, float(NOW.wind_dir), float(NOW.wind), float(NOW.gust)),
                unsafe_allow_html=True,
            )
            head, cross = wind_components(float(NOW.wind_dir), float(NOW.wind), hd)
            if head > 4:
                st.markdown(
                    f"<div class='flag'>You ride out into "
                    f"<b>{head:.0f} mph of headwind</b> and get pushed home. That is the right "
                    f"way round — the hard part happens on fresh legs.</div>",
                    unsafe_allow_html=True)
            elif head < -4:
                st.markdown(
                    f"<div class='flag'>Tailwind out, <b>{abs(head):.0f} mph "
                    f"headwind on the way home</b>, on tired legs. Consider riding this one "
                    f"in reverse, or budget extra time and food for the return.</div>",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<div class='flag flag--ok'>Mostly crosswind at "
                    f"<b>{cross:.0f} mph</b>. Little help either direction, but watch for gusts "
                    f"on exposed bridges and open water.</div>",
                    unsafe_allow_html=True)

        with right:
            eyebrow("Best windows", "one per day, daylight only")
            day = HOURLY[(HOURLY["is_day"] == 1)].copy()
            day["date"] = day["time"].dt.date
            future = day[day["time"] >= pd.Timestamp.now().floor("h")]
            if future.empty:
                future = day
            best = (future.sort_values("score", ascending=False)
                          .groupby("date", as_index=False).head(1)
                          .sort_values("time").head(7))
            wrows = []
            for _, w in best.iterrows():
                pct = float(w["score"])
                col = "#0E7C86" if pct >= 70 else "#E8460F" if pct < 45 else "#6B7580"
                wrows.append(
                    "<tr><td class='day'>{}</td><td>{}{}</td><td>{:.0f}°F</td>"
                    "<td>{:.0f} {}</td><td>{:.0f}%</td></tr>".format(
                        w["time"].strftime("%a %-I%p"), bar_cell(pct, col),
                        score_label(pct), w["temp"], w["wind"],
                        compass_point(w["wind_dir"]), w["precip_prob"]))
            st.markdown(
                "<table class='tt'><thead><tr><th>When</th><th>Riding</th><th>Temp</th>"
                "<th>Wind</th><th>Rain</th></tr></thead><tbody>"
                + "".join(wrows) + "</tbody></table>", unsafe_allow_html=True)
            st.caption(
                "Scored on rain probability, measurable precipitation, wind, gusts and "
                "temperature, weighted so that rain and gusts dominate. Daylight hours only."
            )

    st.divider()
    eyebrow("Routes ranked for today", "weather, shelter and wind on each heading")
    if NOW is not None:
        tbl = []
        for r in VISIBLE:
            hd = r.heading_from(SP)
            h, c = wind_components(float(NOW.wind_dir), float(NOW.wind), hd)
            shelter = 0 if "Sheltered" in r.exposure else (-6 if "Fully exposed" in r.exposure
                                                           or "Brutally" in r.exposure else -3)
            wind_pen = max(0.0, abs(h) + c * 0.5) * (0.0 if "Sheltered" in r.exposure else 0.9)
            row = {
                "Route": r.name,
                "Miles": round(r.total_from(SP), 1),
                "Time": "{}h{:02d}".format(int(r.total_from(SP) / avg_mph),
                                           int(round((r.total_from(SP) / avg_mph % 1) * 60))),
                "Terrain": flatness(r),
                "Traffic": traffic_comfort(r).split("—")[0].strip(),
                "Today": round(max(0.0, NOW.score + shelter - wind_pen), 0),
            }
            if MODE != "Just riding":
                row["Elev ft"] = r.elevation_ft
                row["Wind on route"] = ("head" if h > 2 else "tail" if h < -2 else "cross")
                row["Shelter"] = r.exposure.split("—")[0].strip()
            else:
                row["Coffee"] = "yes" if has_treat(r) else ""
            tbl.append(row)
        df = pd.DataFrame(tbl).sort_values("Today", ascending=False)
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.caption(
            "'Today' adjusts the hour's conditions score for how exposed each route is and "
            "how the wind sits against its outbound heading. A tree-lined park loop loses "
            "almost nothing to a 20 mph wind; an open runway loses a lot."
        )
    else:
        st.dataframe(
            pd.DataFrame([{
                "Route": r.name, "Miles": round(r.total_from(SP), 1),
                "Elev ft": r.elevation_ft, "Shelter": r.exposure.split("—")[0].strip(),
                "Difficulty": r.difficulty,
            } for r in VISIBLE]).sort_values("Miles"),
            hide_index=True, use_container_width=True,
        )


# ============================================================================
# TAB — ROUTE DETAIL
# ============================================================================

CARTO_LIGHT = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
CARTO_DARK = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"


def route_deck(r: Route, sp: StartPoint, basemap: str, radar_tpl: Optional[str],
               show_radar: bool) -> pdk.Deck:
    pts = [(sp.lat, sp.lon)] + r.waypoints
    line = [[lon, lat] for lat, lon in pts]

    layers = []
    if show_radar and radar_tpl:
        layers.append(pdk.Layer("TileLayer", data=radar_tpl, opacity=0.55,
                                min_zoom=0, max_zoom=12, tile_size=256))

    # Casing then stroke gives the route a crisp edge against any basemap
    layers += [
        pdk.Layer("PathLayer", data=pd.DataFrame([{"path": line}]), get_path="path",
                  get_color=[255, 255, 255, 230], width_min_pixels=7, width_scale=1,
                  cap_rounded=True, joint_rounded=True),
        pdk.Layer("PathLayer", data=pd.DataFrame([{"path": line, "label": r.name}]),
                  get_path="path", get_color=[14, 124, 134, 255],
                  width_min_pixels=3.2, cap_rounded=True, joint_rounded=True, pickable=True),
        pdk.Layer("ScatterplotLayer",
                  data=pd.DataFrame([{"lat": la, "lon": lo} for la, lo in r.waypoints]),
                  get_position="[lon, lat]", get_radius=45, radius_min_pixels=2.6,
                  get_fill_color=[14, 124, 134, 210]),
        pdk.Layer("ScatterplotLayer",
                  data=pd.DataFrame([{"lat": s.lat, "lon": s.lon,
                                      "label": s.name, "what": s.what} for s in r.stops]),
                  get_position="[lon, lat]", get_radius=110, radius_min_pixels=7,
                  get_fill_color=[255, 90, 31, 240], get_line_color=[255, 255, 255],
                  stroked=True, line_width_min_pixels=1.6, pickable=True),
        pdk.Layer("ScatterplotLayer",
                  data=pd.DataFrame([{"lat": sp.lat, "lon": sp.lon,
                                      "label": sp.label, "what": "Your start"}]),
                  get_position="[lon, lat]", get_radius=140, radius_min_pixels=8,
                  get_fill_color=[20, 26, 31, 250], get_line_color=[255, 255, 255],
                  stroked=True, line_width_min_pixels=2.2, pickable=True),
    ]

    lats = [p[0] for p in pts] + [s.lat for s in r.stops]
    lons = [p[1] for p in pts] + [s.lon for s in r.stops]
    span = max(max(lats) - min(lats), max(lons) - min(lons))
    zoom = 13.0 if span < 0.04 else 12.0 if span < 0.1 else 11.0 if span < 0.25 else 9.5

    return pdk.Deck(
        map_style=basemap,
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=(min(lats) + max(lats)) / 2,
            longitude=(min(lons) + max(lons)) / 2,
            zoom=zoom, pitch=0, bearing=0,
        ),
        tooltip={
            "html": "<b>{label}</b><br/>{what}",
            "style": {"backgroundColor": "#141A1F", "color": "#E9ECE6",
                      "fontFamily": "IBM Plex Mono, monospace", "fontSize": "11px",
                      "padding": "6px 8px", "borderRadius": "2px"},
        },
    )


with TAB_ROUTE:
    sel = st.selectbox("Route", [r.name for r in VISIBLE], label_visibility="collapsed")
    r = next(x for x in ROUTES if x.name == sel)

    total = r.total_from(SP)
    approach = r.approach_from(SP)
    dur = total / avg_mph
    heading = r.heading_from(SP)

    st.markdown(f"### {r.name}")
    st.markdown(f"<p style='color:#5A6570;margin-top:-0.4rem'>{r.blurb}</p>",
                unsafe_allow_html=True)

    pills = [f"<span class='pill{' pill--on' if r.car_free else ''}'>"
             f"{'car-free' if r.car_free else 'shared roads'}</span>",
             f"<span class='pill'>{r.difficulty}</span>",
             f"<span class='pill'>open road {6 - r.interruption_score}/5</span>"]
    if r.ferry_needed:
        pills.insert(1, "<span class='pill'>ferry</span>")
    st.markdown("<div style='margin:0.2rem 0 0.6rem'>" + "".join(pills) + "</div>",
                unsafe_allow_html=True)

    datastrip([
        ("Total", f"{total:.0f} mi"),
        ("Approach", f"{approach:.1f} mi"),
        ("Elevation", f"{r.elevation_ft:,} ft"),
        ("Moving time", f"{int(dur)}h {int(round((dur % 1) * 60)):02d}m"),
        ("Heading out", f"{heading:.0f}° {compass_point(heading)}"),
    ])

    st.markdown(rows([
        ("Terrain", flatness(r)),
        ("Traffic", traffic_comfort(r)),
        ("Coffee or food on route", "Yes" if has_treat(r) else "Bring your own"),
        ("Good for a beginner", "Yes" if beginner_ok(r) else "Build up to it first"),
    ]), unsafe_allow_html=True)

    mc = st.columns([3, 1])
    with mc[1]:
        basemap_choice = st.radio("Basemap", ["Light", "Dark"], horizontal=True)
        radar_tpl = fetch_radar_frame()
        show_radar = st.toggle("Precipitation radar", value=False,
                               help="Live radar from RainViewer. Nowcast only, not a forecast.")
        if show_radar and not radar_tpl:
            st.caption("Radar unavailable right now.")
        if NOW is not None:
            h, c = wind_components(float(NOW.wind_dir), float(NOW.wind), heading)
            st.markdown(
                rows([
                    ("Outbound", "{} {:.0f} mph".format(
                        "head" if h > 1.5 else "tail" if h < -1.5 else "cross", abs(h))),
                    ("Cross", "{:.0f} mph".format(c)),
                    ("Exposure", r.exposure.split("—")[0].strip()),
                ]), unsafe_allow_html=True)

    with mc[0]:
        st.pydeck_chart(
            route_deck(r, SP, CARTO_DARK if basemap_choice == "Dark" else CARTO_LIGHT,
                       radar_tpl, show_radar),
            use_container_width=True,
        )

    lk = st.columns(3)
    lk[0].link_button(f"Bike directions from {SP.label} ↗", r.gmaps_url(SP),
                      use_container_width=True)
    lk[1].link_button("OpenStreetMap cycle layer ↗",
                      "https://www.openstreetmap.org/#map=12/{:.4f}/{:.4f}&layers=C".format(
                          *r.far_point()), use_container_width=True)
    lk[2].link_button("Windy forecast for this area ↗",
                      "https://www.windy.com/?{:.3f},{:.3f},9".format(*r.far_point()),
                      use_container_width=True)
    st.caption("Black marker is your start, orange markers are stops — hover to read them. "
               "The teal line is a schematic corridor, not a GPS track. Use the directions "
               "link for turn-by-turn.")

    a, b = st.columns([3, 2])
    with a:
        eyebrow("What this ride is")
        st.markdown(r.description)

        eyebrow(f"Getting there from {SP.label}")
        st.markdown(f"<div class='flag flag--ok'>{r.approach.get(start_key, '')}</div>",
                    unsafe_allow_html=True)

        eyebrow("On the route")
        for i, step in enumerate(r.on_route, 1):
            st.markdown(f"{i}. {step}")

        if MODE == "Just riding":
            with st.expander("Tips from people who ride this"):
                for t in r.tips:
                    st.markdown(f"- {t}")
        else:
            eyebrow("Tips")
            for t in r.tips:
                st.markdown(f"- {t}")

    with b:
        eyebrow("The numbers")
        st.markdown(rows(r.stats), unsafe_allow_html=True)

        eyebrow("Conditions")
        st.markdown(rows([
            ("Surface", r.surface),
            ("Interruptions", r.interruptions),
            ("Riding there", r.ride_miles),
            ("Wind exposure", r.exposure),
            ("Good for", ", ".join(r.best_for)),
        ]), unsafe_allow_html=True)

        if r.warnings:
            eyebrow("Watch out")
            for w in r.warnings:
                st.warning(w)

    eyebrow("Stops worth making")
    scols = st.columns(2)
    for i, s in enumerate(r.stops):
        with scols[i % 2]:
            st.markdown(
                f"<div style='padding:0.5rem 0;border-bottom:1px solid #D6DAD4'>"
                f"<span style=\"font-family:'IBM Plex Mono',monospace;font-weight:600;"
                f"font-size:0.86rem\">{s.name}</span>"
                f"<div class='note' style='margin-top:0.15rem'>{s.what}</div></div>",
                unsafe_allow_html=True)


# ============================================================================
# TAB — FORECAST
# ============================================================================

with TAB_WX:
    if not WX["ok"] or HOURLY is None:
        st.error(
            "**Forecast unavailable.** The app could not reach the Open-Meteo service. "
            "This usually means no outbound network access, or the service is briefly down. "
            "Everything else in the app works without it."
        )
        st.caption(f"Detail: {WX['error']}")
    else:
        eyebrow("The week ahead", f"seven days at {SP.label}")

        daily = HOURLY.copy()
        daily["date"] = daily["time"].dt.date
        agg = daily.groupby("date").agg(
            high=("temp", "max"), low=("temp", "min"),
            wind=("wind", "mean"), gust=("gust", "max"),
            rain=("precip_prob", "max"),
            day_score=("score", lambda s: s.nlargest(6).mean()),
        ).reset_index()

        drows = []
        for _, d in agg.iterrows():
            sc = float(d["day_score"])
            col = "#0E7C86" if sc >= 70 else "#E8460F" if sc < 45 else "#6B7580"
            drows.append(
                "<tr><td class='day'>{}</td><td>{}{:.0f}</td><td>{:.0f}/{:.0f}°F</td>"
                "<td>{:.0f} mph</td><td>{:.0f}</td><td>{:.0f}%</td><td>{}</td></tr>".format(
                    pd.Timestamp(d["date"]).strftime("%a %-d %b"), bar_cell(sc, col), sc,
                    d["high"], d["low"], d["wind"], d["gust"], d["rain"],
                    score_label(sc)))
        st.markdown(
            "<table class='tt'><thead><tr><th>Day</th><th>Score</th><th>High / low</th>"
            "<th>Wind</th><th>Gust</th><th>Rain</th><th>Verdict</th></tr></thead><tbody>"
            + "".join(drows) + "</tbody></table>", unsafe_allow_html=True)

        st.caption("Score is the mean of each day's six best daylight hours, 0–100.")

        st.divider()
        eyebrow("Hourly detail")
        pick = st.selectbox("Day", agg["date"].tolist(),
                            format_func=lambda d: pd.Timestamp(d).strftime("%A %d %B"))
        hh = HOURLY[HOURLY["time"].dt.date == pick].copy()
        hh["hour"] = hh["time"].dt.strftime("%H:%M")

        ch1, ch2 = st.columns(2)
        with ch1:
            eyebrow("Wind and gusts, mph")
            st.line_chart(hh.set_index("hour")[["wind", "gust"]],
                          color=["#0E7C86", "#FF5A1F"], height=210)
        with ch2:
            eyebrow("Temperature and rain chance")
            st.line_chart(hh.set_index("hour")[["temp", "precip_prob"]],
                          color=["#141A1F", "#0E7C86"], height=210)

        eyebrow("Wind direction through the day")
        wd = hh[["hour", "wind_dir", "wind", "gust", "temp", "precip_prob", "score"]].copy()
        wd["from"] = wd["wind_dir"].apply(lambda d: f"{compass_point(d)} ({d:.0f}°)")
        wd["rating"] = wd["score"].apply(score_label)
        st.dataframe(
            wd[["hour", "from", "wind", "gust", "temp", "precip_prob", "rating"]].rename(
                columns={"hour": "Hour", "from": "Wind from", "wind": "mph",
                         "gust": "Gust", "temp": "°F", "precip_prob": "Rain %",
                         "rating": "Riding"}),
            hide_index=True, use_container_width=True, height=330,
        )

        st.caption(
            "Forecast from Open-Meteo, refreshed every 30 minutes. Wind direction is the "
            "direction the wind blows **from**, as meteorology reports it — the wind rose on "
            "the Today tab resolves that against your route's heading so you do not have to."
        )


# ============================================================================
# TAB — ZONES & FUELING
# ============================================================================

if SHOW_FUEL:
    with TAB_FUEL:
        eyebrow(f"Your zones · {ZONE_LABEL} · max {max_hr}, resting {rest_hr}, reserve {HRR} bpm")

        zrows = []
        for name, (lo, hi) in ZONES.items():
            mid = (lo + hi) / 2
            p = pct_hrr(mid, max_hr, rest_hr)
            zrows.append({
                "Zone": name,
                "Range": f"{lo}–{hi} bpm",
                "% reserve": f"{pct_hrr(lo, max_hr, rest_hr) * 100:.0f}–"
                             f"{pct_hrr(hi, max_hr, rest_hr) * 100:.0f}%",
                "% max HR": f"{lo / max_hr * 100:.0f}–{hi / max_hr * 100:.0f}%",
                "Fuel mix": f"{carb_fraction_from_hrr(p) * 100:.0f}% carbs",
                "Feels like": ZONE_FEEL[name],
            })
        st.dataframe(pd.DataFrame(zrows), hide_index=True, use_container_width=True)

        # This narrative recomputes from the sidebar every rerun — no fixed numbers.
        hrmax_lo, hrmax_hi = int(round(max_hr * 0.60)), int(round(max_hr * 0.70))
        res_edges = hrr_zone_edges(max_hr, rest_hr)
        res_lo, res_hi = res_edges[0] + 1, res_edges[1]
        overlap_lo, overlap_hi = max(hrmax_lo, res_lo), min(hrmax_hi, res_hi)
        overlap = max(0, overlap_hi - overlap_lo)
        z2_now = ZONES["Z2 · Endurance"]

        if overlap == 0:
            verdict = (f"They do not overlap at all. The entire percent-of-max endurance band "
                       f"({hrmax_lo}–{hrmax_hi}) sits inside your reserve-method **Zone 1**.")
        elif overlap < 6:
            verdict = (f"They overlap by just {overlap} bpm ({overlap_lo}–{overlap_hi}), which is "
                       "not enough to treat them as the same instruction.")
        else:
            verdict = (f"They overlap across {overlap} bpm ({overlap_lo}–{overlap_hi}), so with "
                       "your numbers the two schemes broadly agree.")

        eyebrow("Why the number on your watch may not mean what a coach means")
        st.markdown(
            f"<div class='flag'>"
            f"With <b>max {max_hr}</b> and <b>resting {rest_hr}</b>, the loose percent-of-max "
            f"convention puts endurance riding at <b>{hrmax_lo}–{hrmax_hi} bpm</b>. The heart "
            f"rate reserve method — what Apple Watch uses on Automatic — puts Zone 2 at "
            f"<b>{res_lo}–{res_hi} bpm</b>. {verdict}<br/><br/>"
            f"You are currently reading <b>{ZONE_LABEL}</b>, so Zone 2 for you is "
            f"<b>{z2_now[0]}–{z2_now[1]} bpm</b>. Cross-check it against the talk test: if you "
            f"cannot hold a full conversation, you are above Zone 2 whatever the screen says."
            f"</div>", unsafe_allow_html=True)

        st.divider()
        eyebrow("Fueling for a specific ride")

        fuel_route = st.selectbox("Ride", [x.name for x in VISIBLE], key="fuelroute")
        fr = next(x for x in ROUTES if x.name == fuel_route)
        fdur = fr.total_from(SP) / avg_mph

        zone = st.select_slider("Intensity", options=ZONE_NAMES, value=fr.typical_zone)
        zlo, zhi = ZONES[zone]
        st.caption(f"{zone} on your zones is {zlo}–{zhi} bpm."
                   + ("" if zone == fr.typical_zone else f" Typical for this route is {fr.typical_zone}."))

        fm = fueling_model(fdur, zlo, zhi, max_hr, rest_hr, weight_kg, age, sex,
                           mixed_carb, vo2max)

        datastrip([
            ("Ride time", f"{int(fdur)}h {int(round((fdur % 1) * 60)):02d}m"),
            ("Target HR", f"{fm['hr']} bpm"),
            ("Energy cost", f"{fm['kcal_hr']:.0f} kcal/h"),
            ("Ride total", f"{fm['kcal_total']:.0f} kcal"),
            ("Fuel mix", f"{fm['carb_frac'] * 100:.0f}% carbs"),
        ])

        if fm["kcal_vo2"]:
            st.caption(
                f"Two independent estimates: the HR equation gives {fm['kcal_keytel']:.0f} kcal/h, "
                f"the oxygen-uptake method gives {fm['kcal_vo2']:.0f} kcal/h. Shown is the average. "
                f"The {abs(fm['kcal_keytel'] - fm['kcal_vo2']):.0f} kcal/h spread between them is a "
                "fair picture of the real uncertainty."
            )

        datastrip([
            ("Carbs burned", f"{fm['ox']:.0f} g/h"),
            ("Carbs to eat", f"{fm['intake']:.0f} g/h"),
            ("Total on bike", f"{fm['total_intake']:.0f} g"),
            ("Fluid", f"{fm['fluid_total']:.1f} L"),
        ])

        if fm["intake"] <= 0:
            st.info("**Water is enough.** Under about 75 minutes at this intensity, stored muscle "
                    "glycogen covers the ride. Practising eating on the bike is still worthwhile, "
                    "but it is not required here.")
        else:
            st.markdown(f"**Roughly:** {carb_examples(fm['total_intake'])}")
            if fm["intake"] >= fm["ceiling"] - 0.5:
                extra = (" with mixed glucose and fructose." if mixed_carb else
                         " from a single carbohydrate source. A mixed glucose and fructose product "
                         "raises that to about 90 g/h, though it takes gut training.")
                st.warning(
                    f"**Capped by absorption, not demand.** You burn about {fm['ox']:.0f} g/h here "
                    f"but the gut takes on roughly {fm['ceiling']} g/h{extra} The shortfall comes "
                    "out of stored glycogen, which is why rides at this intensity have a hard time "
                    "limit and why it shows up in the last hour rather than when you made the mistake."
                )
            else:
                st.info(
                    f"You burn about {fm['ox']:.0f} g of carbohydrate an hour here and the target is "
                    f"{fm['intake']:.0f} g/h — deliberately less than you burn. Stored glycogen "
                    "covers part of it; the goal is to slow the drawdown, not match it gram for "
                    "gram. **Start at minute 30 and set a repeating timer.** Hunger arrives 30–45 "
                    "minutes after you actually needed the food."
                )

        with st.expander("Before and after"):
            pre = ("A carb-heavy meal 2–3 hours out, 100–150 g carbs, low fat and low fibre so it "
                   "clears in time." if fdur > 2 else
                   "Something simple 60–90 minutes out: banana and toast with honey, oatmeal, a "
                   "bagel. 40–80 g carbs.")
            st.markdown(
                f"**Before** — {pre}\n\n"
                f"**During** — {fm['intake']:.0f} g carbs and about {fm['fluid_hr']:.1f} L fluid per "
                "hour, starting at minute 30.\n\n"
                f"**After, within about 45 minutes** — around {fm['post_carb']:.0f} g carbs plus "
                f"{fm['post_protein']:.0f} g protein. Scale it to what you burned: this ride comes "
                f"out near {fm['kcal_total']:.0f} kcal."
            )

        with st.expander("How this is calculated, and where it is wrong"):
            cross = (", cross-checked against an oxygen-uptake estimate from your VO2max."
                     if vo2max else ". Add your VO2max in the sidebar for a second estimate.")
            st.markdown(
                f"**The chain**\n\n"
                f"1. Target HR is the midpoint of {zone} on your zones: **{fm['hr']} bpm**, which is "
                f"{fm['pct_hrr'] * 100:.0f}% of your {HRR} bpm reserve.\n"
                f"2. Energy cost from the **Keytel et al. (2005)** heart-rate equation, using HR, "
                f"weight, age and sex{cross}\n"
                f"3. Energy splits into carbohydrate and fat by **% of heart rate reserve**, which "
                "tracks %VO2max. Fat oxidation peaks near 45–65% and carbohydrate takes over above.\n"
                f"4. Intake targets ~60% of oxidation, capped at the gut's absorption limit.\n\n"
                "**Where it is wrong**\n\n"
                "- HR-based calorie estimates carry **10–20% error**, more when max HR is an "
                "age estimate rather than a measurement.\n"
                "- **Cardiac drift** adds 5–10 bpm over a long ride at unchanged effort as you "
                "dehydrate and core temperature climbs. Late-ride HR overstates intensity.\n"
                "- Heat, caffeine, poor sleep and stress all raise HR independently of effort.\n"
                "- On hilly routes HR swings hard either side of your real effort, so a whole-ride "
                "average describes an intensity you never rode.\n"
                "- Substrate percentages are population averages and shift with training.\n"
                "- Resting HR moves with fitness and sleep. Worth re-checking monthly.\n\n"
                "**Treat these as starting points to test against.** The number that matters is "
                "whether you finish rides strong or empty."
            )

        st.caption(
            "General endurance-cycling guidance, not individualised nutrition or medical advice. "
            "Needs vary with body composition, heat, gut tolerance and training history. Test "
            "changes on shorter rides before a big day."
        )

st.divider()
st.caption(
    "Distances and elevation are estimates from route-planning platforms and local sources, "
    "not surveyed measurements. Approach distances for each start are straight-line figures "
    "scaled by a road-detour factor. Conditions change constantly — construction, seasonal "
    "ferry and park-road closures, shop hours and bridge path work. Verify before a long day out. "
    "Weather from Open-Meteo; radar from RainViewer."
)
