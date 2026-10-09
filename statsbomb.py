"""StatsBomb Open Data loader, shared by xg/ and scouting/.

Downloads raw event JSON once into data/events/ (gzipped) and builds tables from it.
Data: https://github.com/statsbomb/open-data (free, attribution required).
"""
import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

RAW = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
DATA = Path(__file__).parent / "data"
EVENTS = DATA / "events"

# The only full-season men's league coverage in the open data (verified: 377-380 matches each).
# Every other season is one team's matches only -> biased.
FULL_SEASONS = {
    "Premier League 2015/16": (2, 27),
    "La Liga 2015/16": (11, 27),
    "Serie A 2015/16": (12, 27),
    "Ligue 1 2015/16": (7, 27),
}


# Some Ligue 1 event files use short names; the matches list uses these full ones.
TEAM_ALIASES = {"Marseille": "Olympique de Marseille", "Caen": "Stade Malherbe Caen"}


def team_name(raw):
    return TEAM_ALIASES.get(raw, raw)


def matches():
    rows = []
    for league, (cid, sid) in FULL_SEASONS.items():
        for m in requests.get(f"{RAW}/matches/{cid}/{sid}.json", timeout=30).json():
            rows.append({"match_id": m["match_id"], "league": league, "date": m["match_date"],
                         "home": m["home_team"]["home_team_name"], "away": m["away_team"]["away_team_name"],
                         "home_score": m["home_score"], "away_score": m["away_score"]})
    return pd.DataFrame(rows)


def _download(match_id):
    path = EVENTS / f"{match_id}.json.gz"
    if path.exists():
        return
    r = requests.get(f"{RAW}/events/{match_id}.json", timeout=60)
    r.raise_for_status()
    tmp = path.with_suffix(".tmp")  # write-then-rename so an interrupted run never leaves a half file
    tmp.write_bytes(gzip.compress(r.content))
    tmp.rename(path)


def download_events(match_ids):
    EVENTS.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(_download, match_ids))


def events(match_id):
    with gzip.open(EVENTS / f"{match_id}.json.gz") as f:
        return json.load(f)


def _shot_rows(match_id):
    rows = []
    evs = events(match_id)
    passes = {e["id"]: e["pass"] for e in evs if e["type"]["name"] == "Pass"}
    for e in evs:
        if e["type"]["name"] != "Shot":
            continue
        s = e["shot"]
        kp = passes.get(s.get("key_pass_id"), {})
        rows.append({
            "match_id": match_id,
            "id": e["id"],
            "period": e["period"],
            "minute": e["minute"],
            "team": e["team"]["name"],
            "player": e["player"]["name"],
            "position": e.get("position", {}).get("name"),
            "x": e["location"][0],
            "y": e["location"][1],
            "body_part": s["body_part"]["name"],
            "shot_type": s["type"]["name"],          # Open Play / Free Kick / Penalty / Corner / Kick Off
            "technique": s["technique"]["name"],
            "play_pattern": e["play_pattern"]["name"],
            "under_pressure": e.get("under_pressure", False),
            "first_time": s.get("first_time", False),
            "one_on_one": s.get("one_on_one", False),
            "assisted": bool(kp),
            "assist_cross": kp.get("cross", False),
            "assist_through_ball": kp.get("through_ball", False),
            "assist_cut_back": kp.get("cut_back", False),
            "assist_height": kp.get("height", {}).get("name"),  # Ground / Low / High Pass
            "freeze_frame": json.dumps(s.get("freeze_frame", [])),
            "outcome": s["outcome"]["name"],
            "is_goal": s["outcome"]["name"] == "Goal",
            "sb_xg": s["statsbomb_xg"],              # benchmark ONLY - never a feature
        })
    return rows


def shots():
    """All shots from the full seasons, cached to data/shots.parquet."""
    out = DATA / "shots.parquet"
    if out.exists():
        return pd.read_parquet(out)
    m = matches()
    download_events(m.match_id)
    df = pd.DataFrame([r for mid in m.match_id for r in _shot_rows(mid)])
    df = df.merge(m[["match_id", "league", "date"]], on="match_id")
    df.to_parquet(out)
    return df


if __name__ == "__main__":
    df = shots()
    print(df.shape)
