"""
NYC Ride Finder — cycling routes from a home base in Brooklyn.

Routes, per-route maps, stops, heart-rate zones, and intensity-aware fueling math.
Zone defaults use the heart rate reserve method, matching Apple Watch.

Run locally:
    pip install -r requirements.txt
    streamlit run streamlit_app.py

Home base is configurable so a public deployment need not expose a street address.
Set HOME_ADDRESS / HOME_LAT / HOME_LON via .streamlit/secrets.toml or environment
variables. Defaults are neighborhood-level only.
"""

import os
import urllib.parse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pydeck as pdk
import streamlit as st


def _setting(key: str, default):
    """Read config from Streamlit secrets, then env, then fall back to default."""
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, default)


# Neighborhood-level default. Override with secrets for a precise start point.
HOME_ADDRESS = str(_setting("HOME_ADDRESS", "Fort Greene, Brooklyn, NY 11205"))
HOME = (float(_setting("HOME_LAT", 40.6893)), float(_setting("HOME_LON", -73.9737)))

st.set_page_config(page_title="NYC Ride Finder", page_icon="🚴", layout="wide")


# ============================================================================
# HEART RATE MODEL
# ============================================================================

ZONE_NAMES = [
    "Z1 · Recovery",
    "Z2 · Endurance",
    "Z3 · Tempo",
    "Z4 · Threshold",
    "Z5 · VO2max",
]

# Zone bands as a fraction of max HR — the loose "%HRmax" cycling convention.
ZONES_HRMAX: Dict[str, Tuple[float, float]] = {
    "Z1 · Recovery":   (0.50, 0.60),
    "Z2 · Endurance":  (0.60, 0.70),
    "Z3 · Tempo":      (0.70, 0.80),
    "Z4 · Threshold":  (0.80, 0.90),
    "Z5 · VO2max":     (0.90, 1.00),
}

# Zone bands as a fraction of lactate threshold HR (Friel convention, simplified).
ZONES_LTHR: Dict[str, Tuple[float, float]] = {
    "Z1 · Recovery":   (0.65, 0.81),
    "Z2 · Endurance":  (0.81, 0.89),
    "Z3 · Tempo":      (0.90, 0.93),
    "Z4 · Threshold":  (0.94, 0.99),
    "Z5 · VO2max":     (1.00, 1.06),
}

# Heart rate reserve boundaries — the method Apple Watch uses on Automatic.
# Zone edges sit at these fractions of (max HR − resting HR), above resting HR.
HRR_BOUNDS = (0.59, 0.69, 0.783, 0.89)

# Share of energy coming from carbohydrate, as a function of % heart rate
# reserve. %HRR tracks %VO2max closely, and substrate use tracks %VO2max — fat
# oxidation peaks around 45–65% and carbohydrate takes over from there up.
_CARB_ANCHORS = [
    (0.30, 0.30), (0.40, 0.35), (0.50, 0.42), (0.60, 0.55),
    (0.70, 0.68), (0.80, 0.80), (0.90, 0.90), (1.00, 0.97),
]


def carb_fraction_from_hrr(pct: float) -> float:
    """Interpolate the carbohydrate share of energy expenditure from %HRR."""
    pct = max(0.20, min(1.05, pct))
    pts = _CARB_ANCHORS
    if pct <= pts[0][0]:
        return pts[0][1]
    if pct >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= pct <= x1:
            t = (pct - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return pts[-1][1]


def fluid_l_per_hour(pct: float) -> float:
    """Rough sweat replacement need in L/hr from %HRR, temperate conditions."""
    return max(0.3, 0.35 + 0.75 * pct)


def tanaka_max_hr(age: int) -> int:
    """Tanaka et al. (2001): 208 − 0.7 × age. More accurate than 220 − age."""
    return int(round(208 - 0.7 * age))


def hrr_zone_edges(max_hr: int, rest_hr: int) -> List[int]:
    """Upper bpm edge of zones 1–4 using the heart rate reserve method."""
    hrr = max(1, max_hr - rest_hr)
    return [int(rest_hr + b * hrr) for b in HRR_BOUNDS]


def build_zones(
    method: str,
    max_hr: int,
    rest_hr: int,
    lthr: int,
    custom_edges: List[int],
) -> Dict[str, Tuple[int, int]]:
    """Return {zone_name: (low_bpm, high_bpm)} for the chosen method."""
    if method == "% of Max HR":
        return {
            n: (int(round(max_hr * lo)), int(round(max_hr * hi)))
            for n, (lo, hi) in ZONES_HRMAX.items()
        }
    if method == "% of LTHR":
        return {
            n: (int(round(lthr * lo)), int(round(lthr * hi)))
            for n, (lo, hi) in ZONES_LTHR.items()
        }
    # Heart rate reserve / Apple Watch style, driven by four zone edges
    e = sorted(custom_edges)
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


def keytel_kcal_per_hour(hr: float, weight_kg: float, age: int, sex: str) -> float:
    """
    Keytel et al. (2005) heart-rate energy expenditure equation.
    Native output is kJ/min; converted here to kcal/hour.
    """
    if sex == "Male":
        kj_min = -55.0969 + (0.6309 * hr) + (0.1988 * weight_kg) + (0.2017 * age)
    else:
        kj_min = -20.4022 + (0.4472 * hr) - (0.1263 * weight_kg) + (0.0740 * age)
    return max(0.0, kj_min / 4.184) * 60


def vo2_kcal_per_hour(
    pct: float, vo2max: float, weight_kg: float, carb_frac: float
) -> float:
    """
    Energy cost via oxygen uptake. %HRR is treated as ≈ %VO2max reserve.
    Caloric equivalent of O2 varies with substrate: ~4.69 kcal/L burning pure
    fat, ~5.05 kcal/L burning pure carbohydrate.
    """
    vo2_rest = 3.5  # ml/kg/min, one MET
    vo2 = vo2_rest + pct * max(0.0, vo2max - vo2_rest)   # ml/kg/min
    litres_min = vo2 * weight_kg / 1000.0
    kcal_per_litre = 4.69 + 0.36 * carb_frac
    return litres_min * kcal_per_litre * 60


def fueling_model(
    duration_hr: float,
    zone_low: int,
    zone_high: int,
    max_hr: int,
    rest_hr: int,
    weight_kg: float,
    age: int,
    sex: str,
    mixed_carb: bool,
    vo2max: Optional[float],
) -> dict:
    """
    Turn duration plus heart-rate intensity into a fueling recommendation.

    1. Target HR is the midpoint of the chosen zone.
    2. Energy cost from the Keytel HR equation, cross-checked against an
       oxygen-uptake estimate when VO2max is known.
    3. Split into carbohydrate and fat using %HRR.
    4. Carb intake targets ~60% of oxidation — stored glycogen covers the rest.
    5. Cap by what the gut can actually absorb.
    6. Short rides need little regardless of intensity.
    """
    hr = (zone_low + zone_high) / 2.0
    pct = pct_hrr(hr, max_hr, rest_hr)
    carb_frac = carb_fraction_from_hrr(pct)

    kcal_keytel = keytel_kcal_per_hour(hr, weight_kg, age, sex)
    kcal_vo2 = (
        vo2_kcal_per_hour(pct, vo2max, weight_kg, carb_frac) if vo2max else None
    )
    kcal_hr = (kcal_keytel + kcal_vo2) / 2 if kcal_vo2 else kcal_keytel

    carb_ox_g_hr = (kcal_hr * carb_frac) / 4.0        # 4 kcal per g carbohydrate
    ceiling = 90 if mixed_carb else 60                # g/hr absorption limit

    intake_g_hr = min(carb_ox_g_hr * 0.60, ceiling)

    # Under ~75 min, stored glycogen covers an easy ride outright
    if duration_hr < 1.25:
        intake_g_hr = 0.0 if pct < 0.70 else min(intake_g_hr, 25)

    fuel_hours = max(0.0, duration_hr - 0.5)          # fuelling starts ~30 min in
    fl_hr = fluid_l_per_hour(pct)

    return {
        "hr": int(round(hr)),
        "pct_hrr": pct,
        "pct_max": hr / max(1, max_hr),
        "kcal_keytel": kcal_keytel,
        "kcal_vo2": kcal_vo2,
        "kcal_hr": kcal_hr,
        "kcal_total": kcal_hr * duration_hr,
        "carb_frac": carb_frac,
        "carb_ox_g_hr": carb_ox_g_hr,
        "intake_g_hr": intake_g_hr,
        "total_intake": intake_g_hr * fuel_hours,
        "ceiling": ceiling,
        "fluid_l_hr": fl_hr,
        "fluid_total": fl_hr * duration_hr,
        "post_carb": kcal_hr * duration_hr * 0.30 / 4.0,
        "post_protein": min(40, max(20, weight_kg * 0.35)),
    }


def carb_examples(grams: float) -> str:
    g = int(round(grams))
    if g <= 5:
        return "—"
    return (
        f"~{max(1, round(g / 22))} gels  ·  "
        f"~{max(1, round(g / 27))} bananas  ·  "
        f"~{max(1, round(g / 18))} medjool dates  ·  "
        f"~{max(1, round(g / 40))} bottles of sports mix"
    )


# ============================================================================
# ROUTE MODEL
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
    approach_miles: float
    loop_miles: str
    total_miles: float
    elevation_ft: int
    surface: str
    interruptions: str
    interruption_score: int          # 1 = wide open, 5 = stoplight every block
    car_free: bool
    ferry_needed: bool
    best_for: List[str]
    difficulty: str
    typical_zone: str
    stats: List[Tuple[str, str]]
    getting_there: List[str]
    description: str
    tips: List[str]
    stops: List[Stop]
    warnings: List[str]
    waypoints: List[Tuple[float, float]]
    gmaps_dest: str
    gmaps_via: List[str] = field(default_factory=list)

    def gmaps_url(self) -> str:
        params = {
            "api": "1",
            "origin": HOME_ADDRESS,
            "destination": self.gmaps_dest,
            "travelmode": "bicycling",
        }
        if self.gmaps_via:
            params["waypoints"] = "|".join(self.gmaps_via)
        return "https://www.google.com/maps/dir/?" + urllib.parse.urlencode(params)

    def osm_url(self) -> str:
        lat, lon = self.waypoints[len(self.waypoints) // 2]
        return f"https://www.openstreetmap.org/#map=13/{lat}/{lon}&layers=C"


ROUTES: List[Route] = [

    Route(
        name="Prospect Park Loop",
        blurb="Your 10-minutes-from-the-door default. Car-free laps, one real hill.",
        approach_miles=1.9, loop_miles="3.35 mi per lap", total_miles=17.0,
        elevation_ft=131,
        surface="Smooth asphalt, dedicated bike lane",
        interruptions="Low — no cars at all, but shared with runners and walkers",
        interruption_score=2, car_free=True, ferry_needed=False,
        best_for=["Endurance / Z2", "Tempo", "Weeknight ride", "Beginner-friendly"],
        difficulty="Easy", typical_zone="Z2 · Endurance",
        stats=[
            ("Loop distance", "3.35 mi — official Prospect Park Alliance figure"),
            ("Elevation per lap", "+131 ft (Ride with GPS)"),
            ("Direction", "Counter-clockwise, one-way only"),
            ("Car-free since", "Jan 2, 2018 — permanently, the entire loop drive"),
            ("Park size", "585 acres"),
            ("Typical lap", "11–13 min at 16–18 mph"),
            ("Where cycling is allowed", "Park Drive, Center Drive, Wellhouse Drive only"),
        ],
        getting_there=[
            "Head south out of Fort Greene to **Lafayette Ave** (a few blocks).",
            "Right on **Lafayette Ave**, then left onto **Vanderbilt Ave** — protected bike lane "
            "most of the way down.",
            "**Vanderbilt Ave** south ~1.4 mi to Grand Army Plaza.",
            "Enter at the Plaza onto **West Drive**, bear right onto **Park Drive** — the "
            "perimeter loop, counter-clockwise.",
        ],
        description=(
            "This is the ride you'll do most. Nothing else in the city gets you onto protected, "
            "car-free pavement this fast — door to first pedal stroke on the loop is realistically "
            "ten minutes.\n\n"
            "The loop is a single 3.35-mile counter-clockwise circuit with one genuine climb on the "
            "east side past the zoo. At +131 ft per lap, six laps (~20 mi) gives you roughly 800 ft "
            "of climbing, respectable for Brooklyn.\n\n"
            "The loop went permanently car-free on January 2, 2018 — before that it had been open "
            "to traffic since the invention of the automobile. Recreational users had been "
            "outnumbering cars roughly three to one, which is why the change stuck."
        ),
        tips=[
            "**Before 8am is a different park.** After about 9am on a weekend the lanes fill with "
            "strollers and runners three-abreast. Early laps are genuinely fast; midday laps are "
            "an exercise in brake modulation.",
            "The bike lane is separate from the pedestrian lane. Stay in it.",
            "**Lap counting:** use the Grand Army Plaza arch as start/finish.",
            "Red lights inside the park are real and enforced. There are only a couple.",
            "Interior pedestrian paths are off-limits to bikes.",
            "**Good HR venue despite the crowds** — the loop is short enough that you can watch "
            "your drift lap to lap and see how much of a rise is fatigue versus dodging.",
        ],
        stops=[
            Stop("Grand Army Plaza", 40.6736, -73.9700,
                 "Soldiers' and Sailors' Arch. De facto start/finish and group-ride meetup."),
            Stop("Picnic House", 40.6690, -73.9720, "Bathrooms and water, mid-park on West Drive."),
            Stop("Prospect Park Lake", 40.6533, -73.9720,
                 "Southern tip of the loop. Brooklyn's only lake."),
            Stop("Bicycle Habitat, 560 Vanderbilt Ave", 40.6800, -73.9690,
                 "Directly on your approach road. Handy for a mid-ride mechanical."),
        ],
        warnings=[
            "Shared lanes with pedestrians make this a poor venue for hard intervals — you'll "
            "spend the effort dodging rather than riding.",
        ],
        waypoints=[HOME, (40.6835, -73.9686), (40.6736, -73.9700), (40.6690, -73.9720),
                   (40.6533, -73.9720), (40.6620, -73.9640), (40.6736, -73.9700)],
        gmaps_dest="Grand Army Plaza, Brooklyn, NY",
    ),

    Route(
        name="Brooklyn Waterfront Greenway",
        blurb="The 30-minute shakeout. Skyline views, cobblestone hazard.",
        approach_miles=1.5, loop_miles="~8 mi of waterfront", total_miles=14.0,
        elevation_ft=120,
        surface="Mixed greenway, protected lane, and DUMBO cobbles",
        interruptions="Moderate — pedestrian-heavy through Brooklyn Bridge Park",
        interruption_score=4, car_free=False, ferry_needed=False,
        best_for=["Recovery spin", "Bike handling", "New-bike shakedown", "Weeknight ride"],
        difficulty="Easy", typical_zone="Z1 · Recovery",
        stats=[
            ("Distance to the water", "1.5 mi"),
            ("Greenway length", "~8 mi, Greenpoint down to Sunset Park"),
            ("Best segment", "Brooklyn Bridge Park waterfront into Columbia St"),
            ("Elevation", "~120 ft — a couple of short ramps, nothing sustained"),
        ],
        getting_there=[
            "Head north to **Navy St** or Flushing Ave.",
            "West through the Navy Yard to **Water St / DUMBO** (~1.5 mi).",
            "Pick up the waterfront path through **Brooklyn Bridge Park**.",
            "South on the **Columbia St** protected lane toward Red Hook and Sunset Park.",
        ],
        description=(
            "Not a training venue so much as a utility ride — your 'I have 45 minutes and don't "
            "want to think about it' option. More importantly, it's where you should do the first "
            "few rides on any new bike.\n\n"
            "The reason: you get cobbles, tight turns, pedestrian chaos, and a couple of short "
            "punchy ramps compressed into a mile and a half from home. If your saddle height is "
            "wrong or your bars are too low, you'll know inside twenty minutes and you'll be close "
            "enough to bail.\n\n"
            "Views are absurd — riding directly beneath the Brooklyn and Manhattan Bridges with "
            "Lower Manhattan across the water."
        ),
        tips=[
            "**DUMBO's cobblestones will rattle your teeth out.** Slow to walking pace on "
            "Washington and Water St, or route around via Front St.",
            "Brooklyn Bridge Park paths are pedestrian-priority and packed on nice evenings.",
            "Red Hook is the quiet reward — wide empty industrial streets, near-zero traffic on "
            "weekend mornings.",
            "Good route for testing lights and reflectives before riding anywhere serious after dark.",
            "**Recovery means recovery.** If your HR won't stay down here, the problem is the "
            "route choice, not your discipline — the stoplights and cobbles spike it whether you "
            "like it or not.",
        ],
        stops=[
            Stop("Redbeard Bikes", 40.7030, -73.9860,
                 "18 Bridge St, DUMBO. Small, community-minded shop known for careful bike fitting."),
            Stop("Brooklyn Roasting Company", 40.7030, -73.9870,
                 "25 Jay St, DUMBO. Longtime cycling-advocacy supporter, bike parking, Dough donuts."),
            Stop("Louis Valentino Jr Pier", 40.6810, -74.0130,
                 "Red Hook. Straight-on Statue of Liberty view, almost always empty."),
            Stop("Brooklyn Bridge Park Pier 6", 40.6930, -74.0000,
                 "Ferry dock, water fountains, bathrooms."),
        ],
        warnings=[
            "Cobblestones plus narrow tires plus pedestrians is how people go down. Ride this one "
            "conservatively.",
        ],
        waypoints=[HOME, (40.6980, -73.9800), (40.7030, -73.9870), (40.6930, -74.0000),
                   (40.6850, -74.0060), (40.6810, -74.0130)],
        gmaps_dest="Louis Valentino Jr. Park and Pier, Brooklyn, NY",
    ),

    Route(
        name="Brooklyn Café & Shop Crawl",
        blurb="A social ride disguised as training. Coffee, shops, and Brooklyn's bike scene.",
        approach_miles=0.0, loop_miles="~22 mi loop", total_miles=22.0,
        elevation_ft=380,
        surface="Protected lanes, greenway, and some rough Bushwick street",
        interruptions="High — this is a city ride with real stoplights",
        interruption_score=5, car_free=False, ferry_needed=False,
        best_for=["Social ride", "Recovery spin", "Weeknight ride", "Sightseeing"],
        difficulty="Easy", typical_zone="Z1 · Recovery",
        stats=[
            ("Loop distance", "~22 mi with all stops"),
            ("Neighborhoods", "Fort Greene → DUMBO → Williamsburg → Bushwick → Bed-Stuy → home"),
            ("Elevation", "~380 ft, mostly the Williamsburg Bridge climb"),
            ("Realistic elapsed time", "3–4 hours with stops, ~1:30 of actual riding"),
            ("Shops on route", "Redbeard, Bicycle Roots, Fulton Bikes, Bicycle Habitat"),
        ],
        getting_there=[
            "Start at the door — this one is a loop from home, no approach.",
            "West to **DUMBO** for **Redbeard Bikes** and **Brooklyn Roasting**.",
            "North along the waterfront through the Navy Yard to **Williamsburg** — "
            "**Domino Park** and **Devoción**.",
            "East on **Grand St** into **Bushwick** for **Sey Coffee**.",
            "South through **Bed-Stuy** — **Bicycle Roots** on Franklin, **Fulton Bikes** on Fulton St.",
            "West on **Fulton St / Lafayette Ave** back to Fort Greene.",
        ],
        description=(
            "Every training plan needs a ride that isn't training. This is that ride.\n\n"
            "A rough 22-mile loop stringing together the best of Brooklyn's cycling and coffee "
            "culture: DUMBO's shops and roasters, the Williamsburg waterfront, Bushwick's "
            "third-wave coffee, and the Bed-Stuy corridor. You'll spend more time off the bike "
            "than on it, and that's the point.\n\n"
            "Practically it's also reconnaissance. You'll meet the shops you'll eventually need — "
            "for a fit adjustment, a wheel true, an emergency tube on a Sunday — and learn which "
            "Brooklyn bike lanes are actually pleasant versus which just exist on a map."
        ),
        tips=[
            "**Bring a lock.** Every other route here you never leave the bike; this one you leave "
            "it constantly. A decent U-lock is non-negotiable in Brooklyn.",
            "Ride it on a weekday morning if you actually want to talk to shop staff.",
            "**Devoción in Williamsburg** roasts fresh-crop Colombian and the space is a "
            "glass-roofed cathedral. Best coffee stop on the loop.",
            "**Sey Coffee in Bushwick** is the other serious one — light roasts, minimalist room, "
            "reliably full of cyclists.",
            "Domino Park's waterfront gives you the skyline framed through the old sugar refinery.",
            "Bushwick's streets are genuinely rough — cobbles, potholes, freight rail crossings.",
            "**Ignore your HR entirely on this one.** Stop-start city riding produces a garbage "
            "average that tells you nothing.",
        ],
        stops=[
            Stop("Redbeard Bikes", 40.7030, -73.9860,
                 "18 Bridge St, DUMBO. Strong reputation for bike fitting and after-sale support."),
            Stop("Brooklyn Roasting Company", 40.7030, -73.9870,
                 "25 Jay St. Bike parking out front, Dough donuts inside."),
            Stop("Domino Park", 40.7145, -73.9680,
                 "Williamsburg waterfront. Skyline framed through the old sugar refinery."),
            Stop("Devoción", 40.7150, -73.9620,
                 "69 Grand St, Williamsburg. Glass-roofed room, fresh-crop Colombian."),
            Stop("Sey Coffee", 40.7070, -73.9330,
                 "18 Grattan St, Bushwick. Light roasts, minimalist space, full of cyclists."),
            Stop("Bicycle Roots", 40.6790, -73.9560,
                 "663 Franklin Ave, Crown Heights. Community-focused, good for repairs."),
            Stop("Fulton Bikes", 40.6830, -73.9600, "997 Fulton St, Bed-Stuy. On your way home."),
        ],
        warnings=[
            "Highest traffic exposure of any route here. Ride defensively, especially crossing "
            "Flushing Ave and along Grand St.",
            "Bike theft is real in these neighborhoods. Never leave a bike unlocked, even briefly.",
        ],
        waypoints=[HOME, (40.7000, -73.9820), (40.7030, -73.9865), (40.7080, -73.9700),
                   (40.7145, -73.9680), (40.7150, -73.9620), (40.7070, -73.9330),
                   (40.6900, -73.9400), (40.6790, -73.9560), (40.6830, -73.9600), HOME],
        gmaps_dest="Devocion, 69 Grand St, Brooklyn, NY",
        gmaps_via=["Redbeard Bikes, 18 Bridge St, Brooklyn, NY", "Domino Park, Brooklyn, NY"],
    ),

    Route(
        name="Rapha Clubhouse Run",
        blurb="Earn the coffee: 14 car-free greenway miles, then the SoHo clubhouse.",
        approach_miles=3.0, loop_miles="~24 mi round trip", total_miles=24.0,
        elevation_ft=320,
        surface="Bridge path, protected lane, and the Hudson River Greenway",
        interruptions="Low on the greenway, high on the SoHo streets",
        interruption_score=3, car_free=False, ferry_needed=False,
        best_for=["Endurance / Z2", "Social ride", "Tempo", "Group riding"],
        difficulty="Easy-Moderate", typical_zone="Z2 · Endurance",
        stats=[
            ("Round trip", "~24 mi"),
            ("Rapha New York", "159 Prince St, at Prince and West Broadway, SoHo"),
            ("Hours", "Roughly 10am–6pm, later Thu–Fri. Verify before planning around it."),
            ("Car-free greenway miles", "~14 of the 24"),
            ("What it is", "Retail plus a full café, home of the NYC Rapha Cycling Club chapter"),
            ("Group rides", "Depart from here; the classic RCC route heads north toward Piermont"),
        ],
        getting_there=[
            "**Manhattan Bridge** — enter at Jay St & Sands St (~1.5 mi from home).",
            "Off the bridge, west on **Canal St** to the **Hudson River Greenway** at the water.",
            "**Greenway north** ~7 mi to around 72nd St — car-free the entire way. This is where "
            "the actual riding happens.",
            "Turn around, **greenway south** back to Canal St.",
            "Cut east a few blocks to **159 Prince St** — Rapha New York.",
            "Home via **Canal St** → **Manhattan Bridge** → Jay St.",
        ],
        description=(
            "The best structure for a café ride: do the work first, then collect the reward.\n\n"
            "The Hudson River Greenway is the single best piece of cycling infrastructure in New "
            "York — a fully separated waterfront path running the length of Manhattan's west side. "
            "Fourteen of this ride's twenty-four miles are on it, which means you can hold a real "
            "steady effort for over an hour without touching a car.\n\n"
            "Then you finish at Rapha New York on Prince Street in SoHo. It's a clubhouse rather "
            "than just a shop: retail up front, a proper café, big screens showing live racing "
            "through the season, and a calendar of talks and events. It's the home of the New York "
            "chapter of the Rapha Cycling Club, and group rides set off from here — the classic one "
            "heads north out of the city toward the rolling roads around Piermont.\n\n"
            "Two-for-one: a genuinely good session, and the most efficient way to plug into the "
            "local road scene from Brooklyn."
        ),
        tips=[
            "**Ride the greenway before the café, not after.** Coffee and a pastry then seven miles "
            "into a headwind is a worse day than you're imagining.",
            "The greenway is busy below 72nd St and much clearer north of there.",
            "**Check the hours before you build the ride around it.** A 24-mile ride to a closed "
            "door is a bad afternoon.",
            "The clubhouse is a legitimate way into group riding. The RCC has a Strava club that "
            "posts rides, and shop rides don't require membership.",
            "Race-viewing parties during the Grand Tours — a three-week excuse to ride to SoHo.",
            "**Extend it:** continue the greenway north to the GWB for a ~40-mile day, or add a "
            "Central Park lap on the way back.",
            "Bring a lock. You're leaving the bike on a SoHo street.",
        ],
        stops=[
            Stop("Rapha New York", 40.7258, -74.0007,
                 "159 Prince St, SoHo. Clubhouse café, retail, live race screenings, and home of "
                 "the NYC Rapha Cycling Club chapter."),
            Stop("Hudson River Greenway", 40.7350, -74.0110,
                 "Around Pier 45. The separated path — the actual training portion of this ride."),
            Stop("Little Island", 40.7415, -74.0100,
                 "Pier 55. The tulip-shaped park on stilts. Free, bikes stay outside."),
            Stop("Little Red Lighthouse", 40.8500, -73.9470,
                 "If you extend north to the GWB. Sits directly under the bridge."),
            Stop("Dinosaur BBQ", 40.8180, -73.9600,
                 "131st & 12th Ave, right off the greenway. Longtime cyclist stop if you go long."),
        ],
        warnings=[
            "The Canal St / SoHo section between the greenway and Prince St is dense, "
            "double-parked, and full of delivery traffic. Slow down for those few blocks.",
            "Hours and even locations of retail change. Confirm the clubhouse is open before "
            "riding out.",
        ],
        waypoints=[HOME, (40.6995, -73.9877), (40.7160, -73.9950), (40.7200, -74.0100),
                   (40.7350, -74.0110), (40.7550, -74.0080), (40.7770, -73.9900),
                   (40.7400, -74.0100), (40.7258, -74.0007), (40.7160, -73.9950), HOME],
        gmaps_dest="Rapha New York, 159 Prince St, New York, NY",
        gmaps_via=["Hudson River Greenway, New York, NY"],
    ),

    Route(
        name="Central Park Loop",
        blurb="6.1 miles, ~390 feet of climbing per lap, and the best hills in Manhattan.",
        approach_miles=5.5, loop_miles="6.1 mi per lap", total_miles=25.0,
        elevation_ft=390,
        surface="Smooth asphalt",
        interruptions="Low in the park, high on the approach",
        interruption_score=3, car_free=True, ferry_needed=False,
        best_for=["Hill repeats", "Tempo", "Endurance / Z2", "Group riding"],
        difficulty="Moderate", typical_zone="Z3 · Tempo",
        stats=[
            ("Loop distance", "6.1 mi (9.8 km)"),
            ("Elevation per lap", "~390 ft (119 m) — three times a Prospect Park lap"),
            ("Harlem Hill", "3.4% average valley-to-crest, 4.4% over its steepest 0.32 mi"),
            ("Cat Hill", "Around E 75th St — gradual, named for the panther statue"),
            ("Named climbs on the loop", "9, of which only Cat and Harlem exceed 5% anywhere"),
            ("Direction", "Counter-clockwise"),
            ("Bail-out", "The 102nd St Crossing lets you skip Harlem Hill entirely"),
        ],
        getting_there=[
            "**Manhattan Bridge** — enter at Jay St & Sands St, ~1.5 mi from home. Take this, not "
            "the Brooklyn Bridge.",
            "Off the bridge, **Canal St** west a few blocks to **Chrystie St**.",
            "**Chrystie St** bike lane north — it feeds into the **2nd Ave protected lane**.",
            "**2nd Ave** north to ~59th St, then west to **Grand Army Plaza** at 5th & 59th.",
            "Enter onto **East Park Drive** and ride counter-clockwise, north up the east side.",
        ],
        description=(
            "The best pure loop in the city, with the caveat that the approach is a tax.\n\n"
            "Six miles per lap with roughly 390 feet of climbing means one Central Park lap has "
            "about three times the vertical of a Prospect Park lap over less than double the "
            "distance. Harlem Hill between 106th and 110th is the real one — 3.4% valley-to-crest "
            "but 4.4% over its steepest third of a mile, and the standard local benchmark. Cat Hill "
            "near 75th is the gentler bookend. Nine named climbs on the loop, but only those two "
            "break five percent anywhere.\n\n"
            "The honest downside: getting there and back burns 70+ minutes of stop-and-go at maybe "
            "10–11 mph. Commuting time, not training time. Which is why Prospect should be your "
            "weeknight default and Central Park a deliberate weekend hill day."
        ),
        tips=[
            "**Take the Manhattan Bridge.** The Brooklyn Bridge path is a tourist obstacle course.",
            "The 2nd Ave protected lane is the best north-south route in Manhattan.",
            "**Harlem Hill repeats:** the north end stays relatively uncrowded even on busy "
            "weekends. If the full loop is packed, do 4–6 repeats of just that segment.",
            "Watch for pedicabs. They stop without warning and don't signal.",
            "Before 7am the loop belongs to fast group rides.",
            "**HR spikes on Harlem Hill and craters on the descent**, so a lap average sits in the "
            "middle and describes neither. Judge the climbs on their own.",
        ],
        stops=[
            Stop("Grand Army Plaza (Manhattan)", 40.7644, -73.9737,
                 "5th Ave & 59th. Main southeast entrance and the standard loop start."),
            Stop("Harlem Hill crest", 40.7970, -73.9580,
                 "Northwest corner, 106th–110th. The park's benchmark climb."),
            Stop("Cat Hill", 40.7745, -73.9660, "East Drive near 75th. The panther statue."),
            Stop("Engineers' Gate", 40.7820, -73.9590,
                 "90th & 5th. Water, bathrooms, and the traditional meeting point."),
            Stop("Rapha New York", 40.7258, -74.0007,
                 "159 Prince St. A 3-mile detour on the way home if you want coffee and a clubhouse."),
        ],
        warnings=[
            "The 5.5 mi approach is not training mileage. The Manhattan crossing will drag your "
            "day's average speed down 3–4 mph — track the laps separately.",
            "Red lights inside the park are ticketed, particularly at the 72nd and 79th St "
            "transverses.",
        ],
        waypoints=[HOME, (40.6995, -73.9877), (40.7160, -73.9950), (40.7231, -73.9928),
                   (40.7460, -73.9760), (40.7644, -73.9737), (40.7745, -73.9660),
                   (40.7970, -73.9580), (40.7900, -73.9620), (40.7644, -73.9737)],
        gmaps_dest="Grand Army Plaza, Central Park South, New York, NY",
        gmaps_via=["Manhattan Bridge Bike Path, Brooklyn, NY"],
    ),

    Route(
        name="Coney Island via Ocean Parkway",
        blurb="America's first bike path, straight to the boardwalk. Dead flat.",
        approach_miles=12.0, loop_miles="Out-and-back", total_miles=24.0,
        elevation_ft=90,
        surface="Ocean Parkway greenway (fair, some root heave), then street",
        interruptions="Moderate-high — cross-street signals the length of Ocean Pkwy",
        interruption_score=4, car_free=False, ferry_needed=False,
        best_for=["Endurance / Z2", "Long ride", "Beginner-friendly"],
        difficulty="Easy", typical_zone="Z2 · Endurance",
        stats=[
            ("Ocean Parkway bike path opened", "1894 — the first dedicated bike path in the US"),
            ("Path length", "~5 mi, Church Ave to Surf Ave"),
            ("Total elevation", "Under 100 ft round trip"),
            ("Round trip from home", "~24 mi, 1:30–1:45 of riding"),
            ("Designed by", "Olmsted and Vaux, same pair as Prospect Park"),
        ],
        getting_there=[
            "Ride to **Prospect Park** via Vanderbilt Ave, then around to the southwest corner at "
            "**Park Circle**.",
            "Pick up the **Ocean Parkway bike path** on the west side of the parkway.",
            "Straight south ~5 mi. You cannot get lost.",
            "Path ends near **Surf Ave** — continue to the **Coney Island boardwalk**, or turn "
            "east toward Brighton Beach.",
        ],
        description=(
            "The simplest long-ish ride in Brooklyn. One straight line south for five miles on a "
            "path that has existed since 1894, when it was built as the first dedicated bicycle "
            "path in the United States — designed by Olmsted and Vaux, the same pair behind "
            "Prospect Park.\n\n"
            "Dead flat, tree-shaded, and the payoff is the Coney Island boardwalk with the Atlantic "
            "in front of you. Round trip lands around 24 miles.\n\n"
            "The tradeoff is signals. Ocean Parkway crosses a numbered street every couple of "
            "blocks. Your average speed will read low and it is not a reflection of your fitness."
        ),
        tips=[
            "**Don't chase average speed here.** The cross-street lights make it structurally "
            "impossible. Duration ride, not a pace ride.",
            "Path surface has root heave and pavement seams — fine on 28mm, jarring on 23mm.",
            "**Extend it:** from Coney Island ride east to Brighton Beach and Sheepshead Bay for "
            "another 4–5 flat miles.",
            "Brighton Beach is the better food stop — Russian bakeries and grocers off Brighton "
            "Beach Ave, cheap and carb-dense.",
            "Afternoon sea breeze means a headwind home. Ride out early.",
            "**Watch HR, not speed.** The lights wreck your pace but your heart rate still tells "
            "you honestly how hard the riding portions were.",
        ],
        stops=[
            Stop("Park Circle", 40.6540, -73.9760, "Southwest corner of Prospect Park. Path starts here."),
            Stop("Coney Island Boardwalk", 40.5730, -73.9800,
                 "Riegelmann Boardwalk, Nathan's, the Cyclone."),
            Stop("Brighton Beach Ave", 40.5780, -73.9600,
                 "Russian bakeries and grocers. Excellent, cheap, carb-dense refueling."),
            Stop("Sheepshead Bay / Emmons Ave", 40.5840, -73.9430,
                 "Fishing boats and seafood. Also the link point toward Rockaway."),
        ],
        warnings=["Boardwalk cycling is time-restricted. Don't assume it's allowed when you arrive."],
        waypoints=[HOME, (40.6736, -73.9700), (40.6540, -73.9760), (40.6250, -73.9720),
                   (40.5990, -73.9700), (40.5780, -73.9600), (40.5730, -73.9800)],
        gmaps_dest="Coney Island Boardwalk, Brooklyn, NY",
        gmaps_via=["Ocean Parkway Bike Path, Brooklyn, NY"],
    ),

    Route(
        name="Shore Parkway Greenway (Bay Ridge)",
        blurb="Best pavement in Brooklyn. Zero cars, harbor on your right the whole way.",
        approach_miles=8.5, loop_miles="~7 mi each way along the harbor", total_miles=30.0,
        elevation_ft=150,
        surface="Smooth separated greenway",
        interruptions="Low once you're on the path",
        interruption_score=2, car_free=True, ferry_needed=False,
        best_for=["Endurance / Z2", "Tempo", "Long ride", "Steady-state efforts"],
        difficulty="Easy-Moderate", typical_zone="Z3 · Tempo",
        stats=[
            ("Greenway length", "~7 mi, Owl's Head Park south and east toward Bensonhurst"),
            ("Approach", "8.5 mi via the 4th Ave protected lane"),
            ("Total elevation", "~150 ft round trip — effectively flat"),
            ("Verrazzano-Narrows Bridge", "693 ft towers, 4,260 ft main span — you ride under it"),
            ("Surface reputation", "Consistently rated the best pavement in Brooklyn by local riders"),
        ],
        getting_there=[
            "South on **Flatbush Ave** to Atlantic, then pick up **4th Ave** — protected bike lane "
            "most of the way.",
            "**4th Ave** south through Park Slope, Gowanus, Sunset Park (~7 mi).",
            "Cut west around 68th St to the **69th St Pier**, or enter at **Owl's Head Park**.",
            "Greenway runs south under the **Verrazzano** and east toward Bath Beach.",
        ],
        description=(
            "The best-kept pavement in the borough. A fully separated greenway along the Narrows "
            "with the harbor on one side and the Belt Parkway on the other — smooth, uninterrupted "
            "surface without ever sharing space with a car.\n\n"
            "Riding directly underneath the Verrazzano-Narrows Bridge is one of those moments that "
            "makes urban cycling worth the hassle. The towers go up 693 feet above you.\n\n"
            "Because it's flat, smooth, and largely free of interruptions, this is one of the few "
            "places in Brooklyn where you can hold a genuine steady effort for 20+ minutes and have "
            "the numbers mean something. It also connects into the Rockaway route."
        ),
        tips=[
            "**4th Ave is the approach.** Not scenic, but the protected lane makes it fast. Avoid "
            "3rd Ave — truck route.",
            "Pedestrian density picks up near the 69th St Pier, then thins dramatically south of "
            "the bridge.",
            "**Best sunset ride in Brooklyn**, no contest.",
            "Owl's Head Park has a short steep climb if you want vertical on a flat day.",
            "Bay Ridge's 5th Ave is a dense strip of bakeries, delis, and Middle Eastern food.",
            "**The best HR-steady venue that doesn't require a 14-mile approach.** Flat and "
            "uninterrupted enough to hold a zone honestly.",
        ],
        stops=[
            Stop("Owl's Head Park", 40.6390, -74.0330, "Short punchy climb, harbor overlook, greenway access."),
            Stop("69th St Pier", 40.6320, -74.0290,
                 "American Veterans Memorial Pier. Ferry to Manhattan, huge harbor views."),
            Stop("Under the Verrazzano", 40.6060, -74.0400, "Directly beneath the towers."),
            Stop("Bay Ridge / 5th Ave", 40.6250, -74.0270,
                 "Bakeries, delis, Middle Eastern food. The refuel strip."),
            Stop("718 Cyclery", 40.6560, -73.9880,
                 "461 7th Ave, South Slope. Custom builds and maintenance workshops."),
        ],
        warnings=[
            "Sections of the Belt Parkway greenway have been under repair periodically. Check for "
            "closures before a long day.",
        ],
        waypoints=[HOME, (40.6800, -73.9770), (40.6620, -73.9910), (40.6450, -74.0100),
                   (40.6390, -74.0330), (40.6320, -74.0290), (40.6060, -74.0400),
                   (40.5990, -74.0100)],
        gmaps_dest="69th Street Pier, Brooklyn, NY",
        gmaps_via=["Owl's Head Park, Brooklyn, NY"],
    ),

    Route(
        name="Floyd Bennett Field",
        blurb="An abandoned airfield. No cars, no lights, no pedestrians, no excuses.",
        approach_miles=14.0, loop_miles="Old runways plus a ~1.4 mi loop road", total_miles=32.0,
        elevation_ft=60,
        surface="Cracked runway concrete plus smooth loop road",
        interruptions="Essentially none — that's the entire point",
        interruption_score=1, car_free=True, ferry_needed=False,
        best_for=["Intervals", "Time trial efforts", "HR / threshold testing", "Aero position practice"],
        difficulty="Moderate (the approach is the work)", typical_zone="Z4 · Threshold",
        stats=[
            ("Opened", "1931 — New York City's first municipal airport"),
            ("Site size", "~1,300 acres, part of Gateway National Recreation Area"),
            ("Straight tarmac available", "Original runways up to roughly a mile"),
            ("Traffic signals on site", "Zero"),
            ("Approach", "14 mi, largely greenway and protected lane"),
            ("Historic note", "Wrong Way Corrigan and Howard Hughes both flew out of here"),
        ],
        getting_there=[
            "Prospect Park → **Ocean Parkway** greenway south (~5 mi).",
            "East on **Avenue U** through Marine Park.",
            "South on **Flatbush Ave** — bike lane and a greenway crossing.",
            "**Floyd Bennett Field** entrance on the left, just before the Marine Parkway Bridge "
            "approach.",
        ],
        description=(
            "The most useful venue on this list, and almost nobody outside the local racing scene "
            "uses it for training.\n\n"
            "Floyd Bennett was NYC's first municipal airport, opened in 1931, now a largely "
            "abandoned 1,300-acre expanse inside Gateway National Recreation Area. Practically: old "
            "runways offering close to a mile of dead-straight, dead-flat tarmac with no cars, no "
            "traffic lights, no pedestrians, and often nobody else at all.\n\n"
            "**This is where you go to get a real threshold heart rate.** Every other venue "
            "contaminates the test — park loops make you brake, city streets make you stop, hills "
            "make HR swing 30 bpm either side of your actual effort. Here nothing interrupts a "
            "20-minute effort, which is exactly what an LTHR test requires."
        ),
        tips=[
            "**The threshold test:** warm up 15 min, then ride 20 min as hard as you can hold "
            "*steadily* — even effort, not a sprint finish. Your average HR over the final 15 of "
            "those 20 minutes approximates your LTHR. Enter it in the sidebar.",
            "**Loop road for smooth sustained efforts, runways for all-out.** The runway concrete "
            "is cracked and seamed.",
            "**Wind is the variable.** Flat coastal plain, nothing to block it. Do out-and-back "
            "efforts, or your numbers are just a wind reading.",
            "Sand drifts across the runways after storms. Scan ahead.",
            "28mm tires minimum. Not a place for 23mm race rubber.",
            "**Pair it with Rockaway.** Floyd Bennett sits right at the foot of the Marine Parkway "
            "Bridge.",
            "Bring everything — no bike shop, no bodega, limited water on site.",
        ],
        stops=[
            Stop("Hangar B", 40.5890, -73.8890,
                 "Volunteer-restored vintage aircraft. Free, limited hours, worth a look."),
            Stop("Floyd Bennett runways", 40.5910, -73.8930, "The old tarmac. Your interval venue."),
            Stop("Aviator Sports", 40.5880, -73.8990,
                 "The only reliable bathroom, water, and vending on the field."),
            Stop("Marine Park Salt Marsh Nature Center", 40.5990, -73.9280,
                 "On the approach. Bathrooms, water, trails."),
        ],
        warnings=[
            "Genuinely remote for NYC. Carry two tubes, a pump, and the ability to fix a flat "
            "yourself.",
            "The 14-mile approach means 32 miles minimum. Don't plan a short session here — and "
            "don't do the LTHR test at the end of it, do it fresh after an easy spin out.",
        ],
        waypoints=[HOME, (40.6736, -73.9700), (40.6540, -73.9760), (40.5990, -73.9700),
                   (40.5990, -73.9280), (40.5920, -73.9020), (40.5895, -73.8888)],
        gmaps_dest="Floyd Bennett Field, Brooklyn, NY",
        gmaps_via=["Ocean Parkway Bike Path, Brooklyn, NY"],
    ),

    Route(
        name="Jamaica Bay Greenway + Shirley Chisholm State Park",
        blurb="Landfill hills with harbor views. Nearly empty on weekdays.",
        approach_miles=10.0, loop_miles="~12 mi of greenway and park trails", total_miles=34.0,
        elevation_ft=250,
        surface="Paved greenway plus some hard-packed trail",
        interruptions="Low once on the greenway",
        interruption_score=2, car_free=True, ferry_needed=False,
        best_for=["Long ride", "Endurance / Z2", "Hill repeats", "Solitude"],
        difficulty="Moderate", typical_zone="Z2 · Endurance",
        stats=[
            ("Shirley Chisholm State Park size", "407 acres"),
            ("Park hours", "9:00 am to dusk — gates close outside that"),
            ("Built on", "The capped Pennsylvania and Fountain Avenue landfills"),
            ("Trails in the park", "~10 mi"),
            ("Opened", "2019 — still relatively unknown"),
            ("Full Jamaica Bay Greenway loop", "~45 mi (72.7 km), ~525 ft climbing"),
        ],
        getting_there=[
            "East on **Atlantic Ave** or Fulton St out of Fort Greene.",
            "South on **Rockaway Ave** toward East New York (~6 mi). Van Sinderen is quieter.",
            "Pick up the **Belt Parkway / Shore Parkway Greenway** heading east.",
            "**Shirley Chisholm State Park** entrance at Pennsylvania Ave.",
        ],
        description=(
            "The most underrated ride within easy reach of Fort Greene.\n\n"
            "Shirley Chisholm State Park is 407 acres built on top of the capped Pennsylvania and "
            "Fountain Avenue landfills — which sounds grim and is the whole reason it's good. "
            "Capping two enormous mounds of garbage produced something Brooklyn otherwise doesn't "
            "have: sustained rolling climbs with unobstructed views over Jamaica Bay and the "
            "Manhattan skyline.\n\n"
            "It opened in 2019 and is still relatively unknown, so on a weekday evening you can "
            "have most of it to yourself. The birdlife is extraordinary — Jamaica Bay is one of the "
            "most important migratory stops on the Atlantic flyway."
        ),
        tips=[
            "**Check the hours.** 9am to dusk, gates locked outside that.",
            "The climbs are short and repeatable — 4–6 repeats of the main mound is a legitimate "
            "hill session without leaving Brooklyn.",
            "Atlantic to Van Sinderen is calmer than Rockaway Ave on the approach.",
            "**Weekday evenings are near-empty.** Weekend mornings bring families.",
            "Water fountains in the park but no food. Bring your own.",
            "The Fountain Ave overlook is the best view in the park and most visitors never get there.",
        ],
        stops=[
            Stop("Shirley Chisholm State Park", 40.6440, -73.8830,
                 "407 acres, 10 mi of trails, real climbs, skyline views. 9am–dusk."),
            Stop("Fountain Ave overlook", 40.6410, -73.8720,
                 "Top of the eastern mound. Best view in the park."),
            Stop("Canarsie Pier", 40.6280, -73.8850,
                 "Gateway NRA pier on Jamaica Bay. Bathrooms, water, seasonal food."),
            Stop("Belt Pkwy Greenway", 40.6350, -73.8900,
                 "Flat separated path along the bay. Connects west to Bay Ridge, east to Rockaway."),
        ],
        warnings=["Some park trails are hard-packed gravel. Fine on 28mm+, sketchy on narrow tires."],
        waypoints=[HOME, (40.6800, -73.9450), (40.6720, -73.9100), (40.6500, -73.8950),
                   (40.6350, -73.8900), (40.6440, -73.8830), (40.6410, -73.8720),
                   (40.6280, -73.8850)],
        gmaps_dest="Shirley Chisholm State Park, Brooklyn, NY",
    ),

    Route(
        name="Liberty State Park (Jersey City)",
        blurb="Flattest, emptiest, smoothest riding in the metro area. Statue on your shoulder.",
        approach_miles=8.0, loop_miles="~6 mi of park paths plus the Jersey City waterfront",
        total_miles=26.0, elevation_ft=80,
        surface="Smooth paths and promenade",
        interruptions="Very low inside the park",
        interruption_score=2, car_free=True, ferry_needed=True,
        best_for=["Intervals", "Tempo", "Endurance / Z2", "HR / threshold testing"],
        difficulty="Easy", typical_zone="Z4 · Threshold",
        stats=[
            ("Park size", "1,212 acres"),
            ("Hudson River Waterfront Walkway", "~2 mi car-free promenade along the park"),
            ("Elevation", "Effectively zero — reclaimed rail yard and fill"),
            ("Ferry", "Liberty Landing Ferry carries bikes; NY Waterway also serves Jersey City"),
            ("Alternative access", "PATH to Newport or Exchange Place, then ride south"),
            ("PATH restriction", "Bikes restricted during weekday rush hours"),
        ],
        getting_there=[
            "**Manhattan Bridge** → Chambers St west (~3 mi).",
            "**Hudson River Greenway** south to **Battery Park** / World Financial Center.",
            "Either the **Liberty Landing Ferry** across (bikes allowed), or **PATH** from WTC to "
            "Newport / Exchange Place.",
            "Jersey side: **Hudson River Waterfront Walkway** south through Paulus Hook → Essex St "
            "→ **Jersey Ave** south → straight onto the pedestrian bridge into the park.",
        ],
        description=(
            "If Floyd Bennett is too far, this is your alternative venue for hard efforts — and "
            "arguably the more pleasant one.\n\n"
            "Liberty State Park is 1,212 acres of reclaimed rail yard on the Jersey City waterfront: "
            "completely flat, wide smooth paths, very little foot traffic away from the Statue of "
            "Liberty viewing area. The two-mile waterfront promenade is car-free and lets you hold "
            "a hard steady effort with Ellis Island and the Statue directly ahead of you.\n\n"
            "The catch is the crossing. You need either a ferry or PATH, both of which add cost, "
            "schedule dependency, and bike restrictions."
        ),
        tips=[
            "**Check the ferry schedule before you leave the house.** Service is seasonal and gaps "
            "can run an hour. PATH restricts bikes at rush hour.",
            "The Jersey City waterfront walkway north of the park adds 4–5 flat car-free miles.",
            "**Second-best HR test venue after Floyd Bennett** — flat and smooth enough that a "
            "20-minute effort stays genuinely steady.",
            "Do clean out-and-back efforts to cancel out wind.",
            "Grove Street in downtown Jersey City has become a serious food neighborhood.",
            "Getting stranded after the last boat means a long detour via Bayonne.",
        ],
        stops=[
            Stop("Liberty State Park", 40.7050, -74.0550,
                 "1,212 acres, flat car-free paths, Statue of Liberty views."),
            Stop("Empty Sky Memorial", 40.7110, -74.0410,
                 "New Jersey's 9/11 memorial. Two walls framing where the towers stood."),
            Stop("Liberty Science Center", 40.7080, -74.0560, "Bathrooms, water, café."),
            Stop("Paulus Hook waterfront", 40.7150, -74.0330,
                 "Car-free walkway with the best straight-on Manhattan view anywhere."),
            Stop("Grove St, Jersey City", 40.7195, -74.0430,
                 "Dense cluster of coffee and food. The refuel stop."),
        ],
        warnings=[
            "Ferry and PATH schedules are the real constraint. Confirm before you commit.",
            "PATH restricts bikes during weekday rush hours.",
        ],
        waypoints=[HOME, (40.6995, -73.9877), (40.7160, -73.9950), (40.7100, -74.0130),
                   (40.7030, -74.0170), (40.7150, -74.0330), (40.7110, -74.0410),
                   (40.7050, -74.0550)],
        gmaps_dest="Liberty State Park, Jersey City, NJ",
    ),

    Route(
        name="Rockaway Beach Round Trip",
        blurb="~44 miles to the Atlantic and back. The classic Brooklyn big day.",
        approach_miles=21.0, loop_miles="Out-and-back with boardwalk options", total_miles=44.0,
        elevation_ft=400,
        surface="Greenway, bike lane, bridge path, boardwalk",
        interruptions="Moderate — one dismount zone, some street sections",
        interruption_score=3, car_free=False, ferry_needed=False,
        best_for=["Long ride", "Endurance / Z2", "Fueling practice", "Big day out"],
        difficulty="Hard (distance, not terrain)", typical_zone="Z2 · Endurance",
        stats=[
            ("Round trip from Fort Greene", "~44 mi"),
            ("Komoot's Rockaway Beach Loop", "42.4 mi (68.2 km), ~394 ft climbing, ~3:06 moving"),
            ("Marine Parkway–Gil Hodges Bridge", "0.75 mi (1.21 km) span, opened 1937"),
            ("Bridge distinction", "Longest vertical-lift bridge open to motor traffic when built"),
            ("Toll for cyclists", "Free"),
            ("Bail-out", "The A train from Rockaway carries bikes"),
        ],
        getting_there=[
            "Prospect Park → **Ocean Parkway** greenway south (~7 mi).",
            "**Brighton Beach**, then east on Emmons Ave to the **Belt Parkway / Shore Parkway "
            "Greenway** past Plumb Beach.",
            "**Flatbush Ave** south — cross at the greenway crossing, don't try to merge.",
            "**Marine Parkway–Gil Hodges Bridge**, bike path on the west side.",
            "Off the bridge you're at **Jacob Riis Park**. Boardwalk riding is legal here.",
            "Cut back to **Rockaway Beach Blvd** — bike lane runs east to B 116th St.",
        ],
        description=(
            "The signature Brooklyn long ride. Roughly 44 miles door to door, almost entirely flat "
            "except for two bridge climbs, ending with the Atlantic Ocean and tacos.\n\n"
            "The route strings together most of the good infrastructure in south Brooklyn: the 1894 "
            "Ocean Parkway path, the Belt Parkway greenway along Jamaica Bay, then the Gil Hodges "
            "Bridge into Queens.\n\n"
            "**This is the ride where fueling stops being optional.** At three hours of moving time "
            "you're well past what stored glycogen covers, and the failure shows up in the last ten "
            "miles rather than when you make the mistake. It's also the ride where a bail-out "
            "matters, and you have a good one: the A train takes bikes back."
        ),
        tips=[
            "**Go early.** The bridge path and Belt greenway both congest midday on weekends, and "
            "the afternoon sea breeze means a headwind home.",
            "**Start eating at minute 30**, not when you feel empty. Hunger lags actual need by "
            "30–45 minutes.",
            "**Do NOT accidentally ride onto the Belt Parkway itself.** The transitions are poorly "
            "signed. If you're on a road with 55 mph traffic, stop and turn around.",
            "The A train back is a legitimate plan, not a failure.",
            "Two tubes. You're a long way from a bike shop for most of this.",
            "**Cardiac drift is the trap on this ride.** Expect your HR to climb 5–10 bpm over "
            "three hours at identical effort as you dehydrate and core temp rises. If you chase the "
            "number down by easing off, you'll finish having ridden much easier than you planned. "
            "Judge by breathing and legs late in the ride, not bpm.",
        ],
        stops=[
            Stop("Brancaccio's Food Shop", 40.6530, -73.9800,
                 "Windsor Terrace, at the top of Ocean Pkwy. Iced coffee and a cheese danish — the "
                 "traditional pre-Rockaway fuel stop."),
            Stop("Plumb Beach", 40.5800, -73.9200,
                 "Greenway waypoint on Jamaica Bay. Horseshoe crabs in spring."),
            Stop("Marine Parkway Bridge", 40.5830, -73.8850,
                 "0.75 mi vertical-lift bridge from 1937. Bike path on the west side, 360° views."),
            Stop("Jacob Riis Park", 40.5680, -73.8770,
                 "Art Deco bathhouse, wide beach, legal boardwalk riding, bathrooms."),
            Stop("Fort Tilden / Battery Harris", 40.5650, -73.8930,
                 "Decommissioned WWII coastal battery. Climb the platform for ocean-to-skyline views."),
            Stop("Rockaway Taco / B 96th St", 40.5850, -73.8130,
                 "The traditional turnaround. Fish tacos on the beach. Tacoway Beach at B 87th is "
                 "the other local pick."),
        ],
        warnings=[
            "**The Marine Parkway Bridge path is narrow and technically a walkway.** Signs ask "
            "cyclists to dismount, and with pedestrians present there is genuinely only inches of "
            "clearance. Early morning or late evening is much safer — and walking it costs you two "
            "minutes.",
            "Sections of the Belt Parkway greenway near Gerritsen Inlet have been under construction "
            "periodically, with a narrow climb through the work zone.",
        ],
        waypoints=[HOME, (40.6736, -73.9700), (40.6540, -73.9760), (40.5990, -73.9700),
                   (40.5780, -73.9600), (40.5840, -73.9430), (40.5800, -73.9200),
                   (40.5895, -73.8888), (40.5830, -73.8850), (40.5680, -73.8770),
                   (40.5810, -73.8370)],
        gmaps_dest="Rockaway Beach Blvd & Beach 116th St, Queens, NY",
        gmaps_via=["Marine Parkway Bridge, Brooklyn, NY"],
    ),

    Route(
        name="River Road / Alpine Hill (Palisades)",
        blurb="Nearly car-free road pinned between the Hudson and the cliffs. Real climbing.",
        approach_miles=17.0, loop_miles="~10 mi, repeatable", total_miles=45.0,
        elevation_ft=1400,
        surface="Good pavement, twisty, some rough patches",
        interruptions="Very low — park road with minimal traffic",
        interruption_score=1, car_free=False, ferry_needed=False,
        best_for=["Hill repeats", "Threshold efforts", "Long ride", "Climbing"],
        difficulty="Hard", typical_zone="Z3 · Tempo",
        stats=[
            ("Henry Hudson Drive", "~11 mi of near-car-free park road along the Palisades"),
            ("Round trip elevation", "~1,400–1,800 ft depending on how much you climb"),
            ("Approach to the GWB", "17 mi, mostly on the car-free Hudson River Greenway"),
            ("GWB bike path", "Free, ~1 mi crossing — south path is the cyclist path"),
            ("State Line Lookout", "~530 ft, the classic summit extension"),
            ("Seasonal", "River Road gates close roughly late fall through early spring"),
        ],
        getting_there=[
            "**Manhattan Bridge** → **2nd Ave** north (or Chrystie → Allen → 1st Ave).",
            "Cut west around 34th St to the **Hudson River Greenway**.",
            "**Hudson River Greenway** north ~9 mi — car-free the whole way.",
            "**George Washington Bridge**, south path, signed for bikes.",
            "Off the bridge, descend to **Henry Hudson Drive (River Road)** and head north.",
        ],
        description=(
            "The local serious-cyclist playground, and the reason NYC riders don't complain more "
            "about living in a flat city.\n\n"
            "Once you're over the George Washington Bridge, Henry Hudson Drive drops you onto a "
            "narrow, twisting park road wedged between the Hudson River and the Palisades cliffs. "
            "Almost no car traffic, decent surface, and it climbs and descends constantly. Alpine "
            "Hill is the standard benchmark ascent back out.\n\n"
            "The approach is pleasant too — the Hudson River Greenway north from Midtown is nine "
            "miles of car-free waterfront path.\n\n"
            "This is a serious day: 45 miles with 1,400+ feet of climbing and long technical "
            "descents. Don't make it your first ride on a new bike."
        ),
        tips=[
            "**Strictly Bicycles** at the foot of the bridge on the Jersey side is a full shop plus "
            "café and the unofficial clubhouse for this ride.",
            "The greenway approach is busy with pedestrians below 72nd St.",
            "**Descents are technical** — tight switchbacks, occasional gravel, real speed. Ride "
            "the first descent conservatively to learn the corners.",
            "**Check that River Road is open** before riding 17 miles to find out.",
            "State Line Lookout is the classic extension if you want more climbing and a summit café.",
            "Layers matter. Cooler on River Road than in the city, and you'll sweat climbing then "
            "freeze descending.",
            "**A whole-ride HR average is meaningless here** — it spikes on climbs and craters on "
            "descents, landing on a number you never actually rode. Look at the climbs alone.",
        ],
        stops=[
            Stop("Little Red Lighthouse", 40.8500, -73.9470,
                 "Under the GWB on the Manhattan side. Worth the short detour."),
            Stop("Strictly Bicycles", 40.8540, -73.9450,
                 "2347 Hudson Terrace, Fort Lee. Shop, café, and the ride's unofficial HQ."),
            Stop("Alpine Boat Basin", 40.9450, -73.9210,
                 "River-level rest stop with the Blackledge-Kearney House. Water and bathrooms."),
            Stop("State Line Lookout", 41.0130, -73.9070,
                 "Top of the Palisades, ~530 ft. Small café. The classic summit."),
            Stop("Dinosaur BBQ", 40.8180, -73.9600,
                 "131st & 12th Ave, right off the greenway. Longtime cyclist favorite."),
        ],
        warnings=[
            "Long technical descents on a road you don't know. First time down, ride it slow.",
            "River Road / Henry Hudson Drive closes seasonally and after storms. Verify before "
            "committing to the 17-mile approach.",
        ],
        waypoints=[HOME, (40.6995, -73.9877), (40.7231, -73.9928), (40.7460, -73.9760),
                   (40.7550, -74.0080), (40.8000, -73.9720), (40.8500, -73.9470),
                   (40.8517, -73.9527), (40.8540, -73.9450), (40.9000, -73.9250),
                   (40.9450, -73.9210)],
        gmaps_dest="Henry Hudson Drive, Alpine, NJ",
        gmaps_via=["George Washington Bridge, New York, NY"],
    ),

    Route(
        name="9W to Nyack",
        blurb="The region's roadie superhighway. 30 miles north on a signed state bike route.",
        approach_miles=17.0, loop_miles="~66 mi round trip from the GWB", total_miles=95.0,
        elevation_ft=2800,
        surface="Wide-shouldered highway designated as a bike route",
        interruptions="Low — 9W has a big shoulder and is built for this",
        interruption_score=2, car_free=False, ferry_needed=False,
        best_for=["Epic long ride", "Endurance / Z2", "Group riding", "Bucket list"],
        difficulty="Very hard (distance)", typical_zone="Z2 · Endurance",
        stats=[
            ("GWB to Nyack", "~30 mi each way"),
            ("Round trip from Central Park", "~66 mi"),
            ("Round trip from Fort Greene", "~95 mi"),
            ("Route designation", "NY State Bike Route 9 — continues 345 mi to Montreal"),
            ("Classic lollipop variant", "44.4 mi / +2,485 ft from the GWB via Bradley-Tweed and Blauvelt"),
            ("Full menu version", "73 mi / ~5,800 ft via Hook Mountain and Rockland Lake"),
        ],
        getting_there=[
            "Same as River Road: **Manhattan Bridge** → 2nd Ave north → **Hudson River Greenway** → "
            "**GWB** (~17 mi).",
            "Off the bridge, follow signs for **NY Bike Route 9** starting on **Hudson Terrace**.",
            "Left onto **E Palisade Ave**, then right onto **Sylvan Way / 9W**.",
            "**9W north ~30 mi to Nyack.** It's signed and it's obvious.",
            "**Coming home:** take **Piermont Road** instead of 9W — prettier, quieter, and 9W "
            "narrows unpleasantly as you approach town.",
        ],
        description=(
            "The ride every NYC cyclist eventually does. 9W is a wide-shouldered highway that "
            "happens to be designated New York State Bike Route 9 — the same route that continues "
            "345 miles north to Montreal — and on a weekend morning it carries so many cyclists "
            "that it functions as an informal peloton.\n\n"
            "It's rolling rather than mountainous: roughly 2,500 feet of climbing over the 66-mile "
            "round trip from the GWB, spread across long gradual grades.\n\n"
            "The honest caveat: 95 miles door-to-door from Fort Greene is a genuine all-day effort. "
            "Two smarter options: shorten the turnaround to **Piermont** (~75 mi total), or ride out "
            "and take **NJ Transit or Metro-North** home.\n\n"
            "The Rapha clubhouse group rides head out this way, incidentally — if you want company "
            "for your first attempt, that's the route in."
        ),
        tips=[
            "**Build to it.** Do several 40–50 mile rides first. The failure mode isn't fitness, "
            "it's being 60 miles from home with nothing left.",
            "**The Runcible Spoon** in Nyack is the traditional turnaround. Closes 6pm.",
            "Piermont is the better turnaround for a shorter day, and **Bunbury's Coffee Shop** "
            "there has been the cyclist stop for decades.",
            "**Fuel every 25–30 minutes from the start.** On a five-hour ride this isn't optional — "
            "see the numbers below, and note the absorption ceiling.",
            "Bike shops on route: Strictly Bicycles at the start, then very little until Nyack.",
            "**Trains take bikes.** NJ Transit and Metro-North are both legitimate exits.",
            "Weekend mornings the shoulder is busy with cyclists. Hold your line and signal.",
            "**Ride the first two hours easier than feels right.** On a five-hour day the discipline "
            "that matters is restraint early, and HR is the honest check on that when fresh legs "
            "are lying to you.",
        ],
        stops=[
            Stop("Strictly Bicycles", 40.8540, -73.9450, "Fort Lee, foot of the GWB. Last real shop."),
            Stop("Bunbury's Coffee Shop", 41.0410, -73.9190,
                 "460 Piermont Ave, Piermont. The decades-old cyclist stop — bike racks full of "
                 "carbon on weekend mornings."),
            Stop("Coffee Ride Café", 41.0800, -73.9200,
                 "South Nyack, in the former village hall at the end of the Cuomo Bridge. Founded "
                 "by cyclists, big bike parking, indoor and patio seating."),
            Stop("The Runcible Spoon", 41.0900, -73.9170,
                 "Nyack. The classic turnaround bakery. Closes 6pm."),
            Stop("Piermont Pier", 41.0400, -73.9080,
                 "Mile-long pier into the Hudson. Worth the detour if the legs allow."),
        ],
        warnings=[
            "95 miles is a full day with real consequences if you bonk far from home. Carry more "
            "food than you think you need and have a train plan.",
            "Hudson River crossings for bikes are limited. Don't improvise a route across the river.",
        ],
        waypoints=[HOME, (40.6995, -73.9877), (40.7460, -73.9760), (40.7550, -74.0080),
                   (40.8517, -73.9527), (40.8540, -73.9450), (40.9200, -73.9500),
                   (41.0000, -73.9350), (41.0410, -73.9190), (41.0908, -73.9179)],
        gmaps_dest="Nyack, NY",
        gmaps_via=["George Washington Bridge, New York, NY"],
    ),

    Route(
        name="Governors Island",
        blurb="2.2 car-free miles in the middle of the harbor. Pure novelty.",
        approach_miles=2.5, loop_miles="2.2 mi car-free loop", total_miles=10.0,
        elevation_ft=40,
        surface="Smooth, fully car-free",
        interruptions="Low, but pedestrian-heavy on weekends",
        interruption_score=3, car_free=True, ferry_needed=True,
        best_for=["Recovery spin", "Social ride", "Sightseeing"],
        difficulty="Easy", typical_zone="Z1 · Recovery",
        stats=[
            ("Island size", "172 acres"),
            ("Perimeter loop", "2.2 mi, entirely car-free"),
            ("Ferry from", "Brooklyn Bridge Park Pier 6, or the Battery Maritime Building"),
            ("Season", "Now open year-round, hours vary seasonally"),
            ("Bikes on ferry", "Permitted, small fee"),
            ("The Hills", "Constructed landform at the south end — small climbs, big panorama"),
        ],
        getting_there=[
            "South to the waterfront, then **Columbia St** to **Brooklyn Bridge Park Pier 6** (~2.5 mi).",
            "Ferry to **Governors Island** — bikes allowed for a small fee.",
            "Ride the 2.2 mi perimeter loop counter-clockwise for the best skyline sequence.",
        ],
        description=(
            "Not training. A nice thing to do on a bike.\n\n"
            "Governors Island is 172 acres in the middle of New York Harbor, ten minutes by ferry "
            "from Brooklyn Bridge Park, with no cars on it at all. The 2.2-mile perimeter loop gives "
            "you Lower Manhattan, the Statue of Liberty, the Verrazzano, and Brooklyn in sequence — "
            "probably the best set of views available from a bicycle anywhere in the city.\n\n"
            "Use it for an easy spin, or when someone's visiting. Don't plan a workout around a "
            "2.2-mile loop shared with families on rented tandems."
        ),
        tips=[
            "**Go on a weekday** if you want to ride rather than weave.",
            "Ferry schedules are limited and the last boat back is early. Check before you go.",
            "Free ferry hours exist on some mornings — worth checking the current schedule.",
            "The Hills at the south end have small climbs and the best harbor panorama.",
            "Food on the island is seasonal and expensive. Bring your own.",
            "Castle Williams (1811) is free to enter and takes ten minutes.",
        ],
        stops=[
            Stop("Pier 6, Brooklyn Bridge Park", 40.6930, -74.0000, "Ferry departure. Bathrooms, water."),
            Stop("The Hills", 40.6860, -74.0200,
                 "Constructed landscape at the south end. Small climbs, huge views."),
            Stop("Castle Williams", 40.6910, -74.0180, "1811 circular fortification. Free to enter."),
            Stop("Colonels Row", 40.6900, -74.0160,
                 "Shaded lawns and old officers' housing. Good rest spot."),
        ],
        warnings=["Seasonal and limited ferry hours. Confirm the last departure before you go."],
        waypoints=[HOME, (40.6950, -73.9850), (40.6930, -74.0000), (40.6895, -74.0170),
                   (40.6860, -74.0200), (40.6910, -74.0180)],
        gmaps_dest="Brooklyn Bridge Park Pier 6, Brooklyn, NY",
    ),
]


# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("🚴 Ride Finder")
st.sidebar.caption(f"Home base: **{HOME_ADDRESS}**")

st.sidebar.divider()
st.sidebar.subheader("You")

age = st.sidebar.number_input("Age", 14, 90, int(_setting("AGE", 30)))
weight_lb = st.sidebar.number_input("Weight (lb)", 80.0, 400.0, float(_setting("WEIGHT_LB", 160.0)), step=1.0)
weight_kg = weight_lb * 0.45359237
sex = st.sidebar.radio("Sex (for the energy equation)", ["Male", "Female"], horizontal=True)

avg_mph = st.sidebar.slider(
    "Average moving speed (mph)", 9.0, 24.0, 15.0, step=0.5,
    help="Drives time and fueling estimates. Be honest — include the slow city sections.",
)

st.sidebar.divider()
st.sidebar.subheader("Heart rate")

max_hr = st.sidebar.number_input(
    "Max HR (bpm)", 120, 220, 191,
    help="A measured or watch-derived value beats any age formula. "
         "value is better.",
)
rest_hr = st.sidebar.number_input("Resting HR (bpm)", 30, 100, int(_setting("REST_HR", 60)))

zone_source = st.sidebar.radio(
    "Zone method",
    ["Heart rate reserve (Apple Watch)", "% of Max HR", "% of LTHR"],
    help="Heart rate reserve is what Apple Watch uses on Automatic. It accounts for your "
         "resting HR, so it's the better default.",
)

lthr = 165
if zone_source == "% of LTHR":
    lthr = st.sidebar.number_input(
        "LTHR (bpm)", 100, 220, 168,
        help="From a 20-min all-out test: average HR over the final 15 minutes.",
    )

auto_edges = hrr_zone_edges(max_hr, rest_hr)
if zone_source == "Heart rate reserve (Apple Watch)":
    with st.sidebar.expander("Fine-tune zone edges", expanded=False):
        st.caption(
            f"Computed from your numbers: {auto_edges}. If your watch shows something "
            "slightly different, type its values here so the app matches your wrist."
        )
        e1 = st.number_input("Z1 tops at", 80, 220, auto_edges[0])
        e2 = st.number_input("Z2 tops at", 80, 220, auto_edges[1])
        e3 = st.number_input("Z3 tops at", 80, 220, auto_edges[2])
        e4 = st.number_input("Z4 tops at", 80, 220, auto_edges[3])
    custom_edges = [e1, e2, e3, e4]
else:
    custom_edges = auto_edges

vo2_known = st.sidebar.checkbox(
    "I know my VO2max / Cardio Fitness", value=False,
    help="Apple Watch reports this as Cardio Fitness. Adding it gives a second, "
         "independent energy estimate to cross-check against.",
)
vo2max = st.sidebar.number_input("VO2max (ml/kg/min)", 20.0, 85.0, 48.0, step=0.5) if vo2_known else None

mixed_carb = st.sidebar.checkbox(
    "Using mixed glucose + fructose fuel", value=False,
    help="Multiple-transportable-carbohydrate products raise the absorption ceiling from "
         "roughly 60 g/hr to around 90 g/hr. Needs gut training.",
)

st.sidebar.divider()
st.sidebar.subheader("Filters")

max_total = st.sidebar.slider("Max total ride (miles)", 10, 100, 100, step=5)
all_purposes = sorted({p for r in ROUTES for p in r.best_for})
purpose = st.sidebar.multiselect("What are you riding for?", all_purposes, default=[])
car_free_only = st.sidebar.checkbox("Car-free riding only")
no_ferry = st.sidebar.checkbox("No ferry required")
low_interrupt = st.sidebar.checkbox("Low-interruption only (for hard efforts)")

st.sidebar.divider()
st.sidebar.caption(
    "Distances and elevation are best estimates from route-planning platforms and local sources. "
    "Map lines are schematic corridors — use the directions link for turn-by-turn."
)

ZONES = build_zones(zone_source, max_hr, rest_hr, lthr, custom_edges)
HRR = max(1, max_hr - rest_hr)


def matches(r: Route) -> bool:
    if r.total_miles > max_total:
        return False
    if purpose and not set(purpose) & set(r.best_for):
        return False
    if car_free_only and not r.car_free:
        return False
    if no_ferry and r.ferry_needed:
        return False
    if low_interrupt and r.interruption_score > 2:
        return False
    return True


filtered = [r for r in ROUTES if matches(r)]


# ============================================================================
# HEADER
# ============================================================================

st.title("NYC Ride Finder")
st.caption("Routes from Fort Greene, Brooklyn — directions, stops, HR zones, and fueling math")

if not filtered:
    st.warning("No routes match those filters. Loosen them up in the sidebar.")
    st.stop()

home_point = pd.DataFrame([{"lat": HOME[0], "lon": HOME[1], "label": f"Home — {HOME_ADDRESS}"}])


# --- zones panel ---------------------------------------------------------
with st.expander(
    f"Your heart rate zones — {zone_source}  ·  max {max_hr}, resting {rest_hr}, "
    f"reserve {HRR} bpm",
    expanded=False,
):
    rows = []
    for name, (lo, hi) in ZONES.items():
        mid = (lo + hi) / 2
        p = pct_hrr(mid, max_hr, rest_hr)
        rows.append({
            "Zone": name,
            "Range": f"{lo}–{hi} bpm",
            "% of reserve": f"{int(round(pct_hrr(lo, max_hr, rest_hr) * 100))}–"
                            f"{int(round(pct_hrr(hi, max_hr, rest_hr) * 100))}%",
            "% of max HR": f"{int(round(lo / max_hr * 100))}–{int(round(hi / max_hr * 100))}%",
            "Carbs vs fat": f"~{int(round(carb_fraction_from_hrr(p) * 100))}% carbohydrate",
            "Feels like": {
                "Z1 · Recovery": "Barely working. Could do this all day.",
                "Z2 · Endurance": "Full conversation possible. The bread and butter.",
                "Z3 · Tempo": "Short sentences only. Comfortably hard.",
                "Z4 · Threshold": "A few words. Sustainable ~20–60 min.",
                "Z5 · VO2max": "No talking. Minutes, not hours.",
            }[name],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("**The conventions disagree, and it matters**")
    hrmax_z2 = (int(round(max_hr * 0.60)), int(round(max_hr * 0.70)))
    hrr_edges = hrr_zone_edges(max_hr, rest_hr)
    st.markdown(
        f"With your numbers, the loose **%HRmax** convention puts endurance riding at "
        f"**{hrmax_z2[0]}–{hrmax_z2[1]} bpm**. **Heart rate reserve** — what your watch uses — "
        f"puts Zone 2 at **{hrr_edges[0] + 1}–{hrr_edges[1]} bpm**.\n\n"
        f"Those barely overlap. The entire %HRmax 'endurance zone' sits inside your watch's "
        f"**Zone 1**. So 'ride in Zone 2' means two completely different efforts depending on "
        f"which scheme you're reading, and the gap lands exactly where you'd spend most of your "
        f"time.\n\n"
        f"**The practical resolution:** use the reserve method, because it's what your watch "
        f"shows you mid-ride and it accounts for your resting HR. Then sanity-check it with the "
        f"talk test — if you can't hold a full conversation, you're above Zone 2 no matter what "
        f"the screen says."
    )
    st.caption(
        "Zone boundaries are conventions, not physiology. If you've measured LTHR from a "
        "20-minute test, that's a more accurate anchor than any age-based estimate — and it "
        "usually lands close to the reserve-method zones."
    )

st.subheader("All routes")

overview_paths = pd.DataFrame([
    {"label": f"{r.name} — {r.total_miles:.0f} mi", "path": [[lon, lat] for lat, lon in r.waypoints]}
    for r in filtered
])
overview_points = pd.DataFrame([
    {"lat": r.waypoints[-1][0], "lon": r.waypoints[-1][1],
     "label": f"{r.name} — {r.total_miles:.0f} mi, {r.elevation_ft:,} ft"}
    for r in filtered
])

st.pydeck_chart(pdk.Deck(
    map_style=None,
    layers=[
        pdk.Layer("PathLayer", data=overview_paths, get_path="path",
                  get_color=[70, 130, 200, 150], width_min_pixels=2, pickable=True),
        pdk.Layer("ScatterplotLayer", data=overview_points, get_position="[lon, lat]",
                  get_radius=400, radius_min_pixels=6, get_fill_color=[215, 85, 55, 210],
                  pickable=True),
        pdk.Layer("ScatterplotLayer", data=home_point, get_position="[lon, lat]",
                  get_radius=500, radius_min_pixels=8, get_fill_color=[30, 100, 210, 240],
                  pickable=True),
    ],
    initial_view_state=pdk.ViewState(latitude=40.78, longitude=-73.95, zoom=8.7),
    tooltip={"text": "{label}"},
))
st.caption("Blue dot is home. Red dots are route endpoints. Hover for details.")

with st.expander("Compare all routes side by side"):
    st.dataframe(
        pd.DataFrame([
            {
                "Route": r.name,
                "Total mi": r.total_miles,
                "Elev ft": r.elevation_ft,
                "Est. time": f"{r.total_miles / avg_mph:.1f} h",
                "Typical zone": r.typical_zone.split(" · ")[0],
                "Open road": "▮" * (6 - r.interruption_score) + "▯" * (r.interruption_score - 1),
                "Car-free": "✓" if r.car_free else "",
                "Ferry": "✓" if r.ferry_needed else "",
                "Difficulty": r.difficulty,
            }
            for r in sorted(filtered, key=lambda x: x.total_miles)
        ]),
        hide_index=True, use_container_width=True,
    )
    st.caption(
        "Open road: more bars = fewer interruptions. Five is an abandoned airfield; one is a "
        "stoplight every couple of blocks."
    )


# ============================================================================
# ROUTE DETAIL
# ============================================================================

st.divider()
st.subheader("Click into a route")

names = [r.name for r in sorted(filtered, key=lambda x: x.total_miles)]
choice = st.selectbox("Route", names, label_visibility="collapsed")
r = next(x for x in ROUTES if x.name == choice)

st.markdown(f"## {r.name}")
st.markdown(f"*{r.blurb}*")

dur = r.total_miles / avg_mph
m = st.columns(5)
m[0].metric("Total distance", f"{r.total_miles:.0f} mi")
m[1].metric("Approach", f"{r.approach_miles:.1f} mi")
m[2].metric("Elevation", f"{r.elevation_ft:,} ft")
m[3].metric("Est. moving time", f"{int(dur)}h {int(round((dur % 1) * 60)):02d}m")
m[4].metric("Difficulty", r.difficulty)

# --- per-route map ---
st.markdown("#### Map")

path_df = pd.DataFrame([{"label": r.name, "path": [[lon, lat] for lat, lon in r.waypoints]}])
stops_df = pd.DataFrame([{"lat": s.lat, "lon": s.lon, "label": f"{s.name} — {s.what}"} for s in r.stops])
node_df = pd.DataFrame([{"lat": lat, "lon": lon} for lat, lon in r.waypoints])

lats = [lat for lat, _ in r.waypoints] + [s.lat for s in r.stops] + [HOME[0]]
lons = [lon for _, lon in r.waypoints] + [s.lon for s in r.stops] + [HOME[1]]
span = max(max(lats) - min(lats), max(lons) - min(lons))
zoom = 13.0 if span < 0.05 else 12.0 if span < 0.12 else 10.8 if span < 0.3 else 9.4

st.pydeck_chart(pdk.Deck(
    map_style=None,
    layers=[
        pdk.Layer("PathLayer", data=path_df, get_path="path",
                  get_color=[70, 130, 200, 220], width_min_pixels=4, pickable=True),
        pdk.Layer("ScatterplotLayer", data=node_df, get_position="[lon, lat]",
                  get_radius=90, radius_min_pixels=3, get_fill_color=[70, 130, 200, 190]),
        pdk.Layer("ScatterplotLayer", data=stops_df, get_position="[lon, lat]",
                  get_radius=170, radius_min_pixels=7, get_fill_color=[240, 160, 40, 235],
                  pickable=True),
        pdk.Layer("ScatterplotLayer", data=home_point, get_position="[lon, lat]",
                  get_radius=200, radius_min_pixels=8, get_fill_color=[30, 100, 210, 240],
                  pickable=True),
    ],
    initial_view_state=pdk.ViewState(
        latitude=(min(lats) + max(lats)) / 2,
        longitude=(min(lons) + max(lons)) / 2,
        zoom=zoom,
    ),
    tooltip={"text": "{label}"},
))

lk = st.columns(2)
lk[0].link_button("Turn-by-turn bike directions ↗", r.gmaps_url(), use_container_width=True)
lk[1].link_button("Open in OpenStreetMap (cycle layer) ↗", r.osm_url(), use_container_width=True)
st.caption("Orange dots are stops — hover to read them. Blue line is the route corridor, not a GPS track.")

left, right = st.columns([3, 2])

with left:
    st.markdown("#### What this ride is")
    st.markdown(r.description)
    st.markdown("#### Getting there from home")
    for i, step in enumerate(r.getting_there, 1):
        st.markdown(f"{i}. {step}")
    st.markdown("#### Tips")
    for t in r.tips:
        st.markdown(f"- {t}")

with right:
    st.markdown("#### The numbers")
    for label, val in r.stats:
        st.markdown(f"**{label}** · {val}")
    st.markdown("#### Conditions")
    st.markdown(f"**Surface** · {r.surface}")
    st.markdown(f"**Interruptions** · {r.interruptions}")
    st.markdown(f"**Riding once there** · {r.loop_miles}")
    st.markdown(f"**Car-free** · {'Yes' if r.car_free else 'No'}")
    st.markdown(f"**Ferry needed** · {'Yes' if r.ferry_needed else 'No'}")
    st.markdown("**Good for** · " + ", ".join(r.best_for))
    if r.warnings:
        st.markdown("#### Watch out")
        for w in r.warnings:
            st.warning(w)

st.markdown("#### Stops worth making")
for s in r.stops:
    st.markdown(f"- **{s.name}** — {s.what}")


# ============================================================================
# FUELING
# ============================================================================

st.markdown("#### Fueling — driven by heart rate, not just distance")

zone = st.select_slider(
    "How hard are you riding this?", options=ZONE_NAMES, value=r.typical_zone,
)
zlo, zhi = ZONES[zone]
st.caption(
    f"**{zone}** on your zones is **{zlo}–{zhi} bpm**"
    + (f". Typical for this route is {r.typical_zone}." if zone != r.typical_zone else ".")
)

fm = fueling_model(dur, zlo, zhi, max_hr, rest_hr, weight_kg, age, sex, mixed_carb, vo2max)

c = st.columns(4)
c[0].metric("Target HR", f"{fm['hr']} bpm",
            help=f"{fm['pct_hrr'] * 100:.0f}% of reserve, {fm['pct_max'] * 100:.0f}% of max")
c[1].metric("Energy cost", f"{fm['kcal_hr']:.0f} kcal/hr")
c[2].metric("Total for the ride", f"{fm['kcal_total']:.0f} kcal")
c[3].metric("Fuel mix", f"{fm['carb_frac'] * 100:.0f}% carbs")

if fm["kcal_vo2"]:
    st.caption(
        f"Two independent estimates: HR equation says {fm['kcal_keytel']:.0f} kcal/hr, "
        f"oxygen-uptake method says {fm['kcal_vo2']:.0f} kcal/hr. Shown above is the average. "
        f"The spread between them ({abs(fm['kcal_keytel'] - fm['kcal_vo2']):.0f} kcal/hr) is a "
        "fair picture of the real uncertainty."
    )

st.markdown("**What to actually eat on the bike**")
f = st.columns(4)
f[0].metric("Carbs burned", f"{fm['carb_ox_g_hr']:.0f} g/hr")
f[1].metric("Carbs to eat", f"{fm['intake_g_hr']:.0f} g/hr")
f[2].metric("Total on the bike", f"{fm['total_intake']:.0f} g")
f[3].metric("Fluid", f"{fm['fluid_total']:.1f} L")

if fm["intake_g_hr"] <= 0:
    st.info(
        "**Water is enough for this one.** Under about 75 minutes at this intensity, stored "
        "muscle glycogen covers the whole ride. Eating on the bike is still a skill worth "
        "practicing, but it isn't a requirement here."
    )
else:
    st.markdown(f"**Roughly:** {carb_examples(fm['total_intake'])}")
    if fm["intake_g_hr"] >= fm["ceiling"] - 0.5:
        st.warning(
            f"**Capped by absorption, not by demand.** You're burning about "
            f"{fm['carb_ox_g_hr']:.0f} g/hr but the gut only takes on roughly {fm['ceiling']} g/hr"
            + (" with mixed glucose + fructose." if mixed_carb else
               " from a single carbohydrate source. Switching to a mixed glucose + fructose "
               "product raises that ceiling to about 90 g/hr, though it takes gut training.")
            + " The shortfall comes out of stored glycogen — which is exactly why rides at this "
              "intensity have a hard time limit, and why the last hour is where it shows up."
        )
    else:
        st.info(
            f"You burn about {fm['carb_ox_g_hr']:.0f} g of carbohydrate per hour here, and the "
            f"target is roughly {fm['intake_g_hr']:.0f} g/hr — deliberately less than you burn. "
            "Stored glycogen is meant to cover part of it; the goal is to slow the drawdown, not "
            "match it gram for gram. **Start at minute 30 and set a repeating timer** — hunger "
            "shows up 30–45 minutes after you actually needed the food."
        )

with st.expander("Before and after"):
    pre = (
        "A normal carb-heavy meal 2–3 hours out, 100–150 g carbs, low fat and low fiber so it "
        "clears in time."
        if dur > 2 else
        "Something simple 60–90 min out: banana and toast with honey, oatmeal, a bagel. 40–80 g carbs."
    )
    st.markdown(
        f"**Before** — {pre}\n\n"
        f"**During** — {fm['intake_g_hr']:.0f} g carbs/hr and roughly {fm['fluid_l_hr']:.1f} L "
        "fluid/hr, starting at minute 30.\n\n"
        f"**After (within ~45 min)** — around {fm['post_carb']:.0f} g carbs plus "
        f"{fm['post_protein']:.0f} g protein. Scale to what you burned: this ride comes out "
        f"around {fm['kcal_total']:.0f} kcal."
    )

with st.expander("How this is calculated, and where it's wrong"):
    st.markdown(
        f"**The chain**\n"
        f"1. Target HR is the midpoint of {zone} on your zones: **{fm['hr']} bpm**, which is "
        f"{fm['pct_hrr'] * 100:.0f}% of your {HRR} bpm heart rate reserve.\n"
        f"2. Energy cost from the **Keytel et al. (2005)** HR equation (HR, weight, age, sex), "
        "validated on steady moderate-to-vigorous exercise"
        + (", cross-checked against an oxygen-uptake estimate from your VO2max." if vo2max else
           ". Add your VO2max in the sidebar for a second independent estimate.") + "\n"
        "3. Energy splits into carbohydrate and fat based on **% of heart rate reserve**, which "
        "tracks %VO2max closely. Fat oxidation peaks around 45–65% and carbohydrate takes over "
        "above that.\n"
        "4. Carbohydrate intake targets ~60% of oxidation, then gets capped at the gut's "
        "absorption limit.\n\n"
        "**Where it's wrong**\n"
        "- HR-based calorie estimates typically carry **10–20% error**, more if max HR is an "
        "estimate rather than a measurement.\n"
        "- **Cardiac drift** inflates HR by 5–10 bpm over a long ride at unchanged effort as you "
        "dehydrate and core temperature climbs. Late-ride HR overstates intensity.\n"
        "- Heat, caffeine, poor sleep, altitude, and stress all push HR up independently of effort.\n"
        "- On hilly routes HR swings hard either side of your actual effort, so a whole-ride "
        "average describes a intensity you never rode.\n"
        "- Substrate percentages are population averages. Individual metabolic flexibility varies "
        "a lot and shifts with training.\n"
        "- Resting HR itself moves with fitness, sleep, and stress — worth re-checking monthly, "
        "which is roughly what your watch does on its own.\n\n"
        "**Treat these as starting points to test against.** The number that actually matters is "
        "whether you finish rides feeling strong or empty."
    )

st.caption(
    "General endurance-cycling guidance, not individualized nutrition or medical advice. Real "
    "needs vary with body composition, heat, gut tolerance, and training history — test changes "
    "on shorter rides before a big day."
)

st.divider()
st.caption(
    "Conditions around NYC change constantly: construction, seasonal ferry and park-road closures, "
    "shop and café hours, and bridge path work. Verify before a long day out."
)
