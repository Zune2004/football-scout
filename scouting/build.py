"""Project 2, step 1: parse every match once into player-match and team-match tables.
Outputs data/player_match.parquet and data/team_match.parquet. Run: .venv/Scripts/python scouting/build.py
StatsBomb orients every event so the acting team attacks towards x=120."""
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from statsbomb import DATA, events, matches, team_name

OPEN_PLAY = {None, "Recovery", "Interception"}       # pass types that aren't set pieces
ON_BALL = {"Pass", "Carry", "Shot", "Dribble"}        # actions that show where a player has the ball
DEF_ACTIONS = {"Duel", "Interception", "Ball Recovery", "Block", "Clearance", "Pressure"}
RED = {"Red Card", "Second Yellow"}
_XG = None


def xg_lookup():
    global _XG  # loaded once per worker process
    if _XG is None:
        f = DATA / "shot_xg.parquet"   # our own xG (football-xg-lab repo); else StatsBomb's per-shot xG
        _XG = (pd.read_parquet(f).set_index("id").xg if f.exists()
               else pd.read_parquet(DATA / "shots.parquet").set_index("id").sb_xg).to_dict()
    return _XG


def to_goal(x, y):
    return np.hypot(120 - x, 40 - y)


def progressive(start, end):
    """Wyscout-style: ball moved closer to goal by >=30 units within own half, >=15 crossing halfway, >=10 in opp half.
    ponytail: fixed thresholds; FBref's 'furthest point of last 6 passes' rule is closer to scouting tools if needed."""
    gain = to_goal(*start) - to_goal(*end)
    if start[0] < 60 and end[0] < 60:
        return gain >= 30
    if start[0] < 60:
        return gain >= 15
    return gain >= 10


def in_box(loc):
    return loc[0] >= 102 and 18 <= loc[1] <= 62


def parse_match(match_id):
    ev = events(match_id)
    xg = xg_lookup()
    end = max(e["minute"] + e["second"] / 60 for e in ev)
    on, off, team_of, pos_counts = {}, {}, {}, defaultdict(Counter)
    p = defaultdict(Counter)                      # player id -> stat counter
    t = defaultdict(Counter)                      # team name -> stat counter
    names = {}
    teams = sorted({team_name(e["team"]["name"]) for e in ev})

    for e in ev:
        typ, team, ts = e["type"]["name"], team_name(e["team"]["name"]), e["minute"] + e["second"] / 60
        if typ == "Starting XI":
            for pl in e["tactics"]["lineup"]:
                on[pl["player"]["id"]] = 0.0
                team_of[pl["player"]["id"]] = team
                names[pl["player"]["id"]] = pl["player"]["name"]
            continue
        if typ == "Substitution":
            off[e["player"]["id"]] = ts
            rep = e["substitution"]["replacement"]
            on[rep["id"]], team_of[rep["id"]], names[rep["id"]] = ts, team, rep["name"]
        for k in ("foul_committed", "bad_behaviour"):
            if e.get(k, {}).get("card", {}).get("name") in RED:  # earliest exit wins: a subbed-off player can be sent off from the bench
                off[e["player"]["id"]] = min(off.get(e["player"]["id"], ts), ts)

        pid = e.get("player", {}).get("id")
        s, tm = p[pid], t[team]
        if pid is not None and "position" in e:
            pos_counts[pid][e["position"]["name"]] += 1
        loc = e.get("location")

        if typ in ON_BALL and loc:
            s["actions"] += 1
            s["action_x"] += loc[0]
            s["action_width"] += abs(loc[1] - 40)
            s["actions_att_third"] += loc[0] >= 80
            s["actions_def_third"] += loc[0] < 40
            s["actions_in_box"] += in_box(loc)

        if typ == "Pass":
            pa = e["pass"]
            if pa.get("type", {}).get("name") not in OPEN_PLAY:
                tm["set_piece_passes"] += 1
                continue
            done = "outcome" not in pa
            endl = pa["end_location"]
            s["passes"] += 1; s["passes_cmp"] += done
            s["pass_length"] += pa["length"]
            long_ = pa["length"] >= 30
            s["long_passes"] += long_; s["long_passes_cmp"] += long_ and done
            s["crosses"] += pa.get("cross", False); s["crosses_cmp"] += pa.get("cross", False) and done
            s["through_balls"] += pa.get("through_ball", False)
            s["switches"] += pa.get("switch", False)
            s["aerials_won"] += pa.get("aerial_won", False)
            if "assisted_shot_id" in pa:
                s["key_passes"] += 1
                s["xa"] += xg.get(pa["assisted_shot_id"], 0.0)
            s["assists"] += pa.get("goal_assist", False)
            if done:
                s["prog_passes"] += progressive(loc, endl)
                s["passes_final_third"] += loc[0] < 80 <= endl[0]
                s["passes_into_box"] += in_box(endl) and not in_box(loc)
            tm["passes"] += 1; tm["passes_cmp"] += done; tm["pass_length"] += pa["length"]; tm["long_passes"] += long_
            tm["crosses"] += pa.get("cross", False)
            if done and loc[0] < 80 <= endl[0]:
                tm["final_third_entries"] += 1
                tm["central_entries"] += 80 / 3 <= endl[1] <= 160 / 3
            tm["passes_own_60"] += loc[0] < 72                      # for the OPPONENT's PPDA
        elif typ == "Carry":
            c_end = e["carry"]["end_location"]
            if progressive(loc, c_end):
                s["prog_carries"] += 1
            s["carries_into_box"] += in_box(c_end) and not in_box(loc)
            if loc[0] < 80 <= c_end[0]:
                tm["final_third_entries"] += 1
                tm["central_entries"] += 80 / 3 <= c_end[1] <= 160 / 3
        elif typ == "Dribble":
            s["dribbles"] += 1; s["dribbles_won"] += e["dribble"]["outcome"]["name"] == "Complete"
        elif typ == "Shot":
            sh = e["shot"]
            s["aerials_won"] += sh.get("aerial_won", False)
            if sh["type"]["name"] == "Penalty":
                continue
            x = xg.get(e["id"], 0.0)
            goal = sh["outcome"]["name"] == "Goal"
            s["shots"] += 1; s["np_xg"] += x; s["np_goals"] += goal
            s["shots_on_target"] += sh["outcome"]["name"] in ("Goal", "Saved", "Saved to Post")
            tm["shots"] += 1; tm["np_xg"] += x; tm["np_goals"] += goal
            tm["counter_shots"] += e["play_pattern"]["name"] == "From Counter"
        elif typ == "Duel":
            d = e["duel"]
            if d["type"]["name"] == "Aerial Lost":
                s["aerials_lost"] += 1
            else:
                s["tackles"] += 1
                s["tackles_won"] += d.get("outcome", {}).get("name") in ("Won", "Success In Play", "Success Out")
        elif typ == "Clearance":
            s["clearances"] += 1; s["aerials_won"] += e["clearance"].get("aerial_won", False)
        elif typ == "Miscontrol":
            s["miscontrols"] += 1; s["aerials_won"] += e.get("miscontrol", {}).get("aerial_won", False)
        elif typ == "Interception":
            s["interceptions"] += 1
        elif typ == "Pressure":
            s["pressures"] += 1; s["pressures_att_third"] += loc[0] >= 80
            tm["pressures"] += 1; tm["pressures_att_third"] += loc[0] >= 80
        elif typ == "Ball Recovery":
            s["recoveries"] += 1
        elif typ == "Block":
            s["blocks"] += 1
        elif typ == "Dispossessed":
            s["dispossessed"] += 1
        elif typ == "Dribbled Past":
            s["dribbled_past"] += 1
        elif typ == "Foul Committed":
            s["fouls"] += 1

        if typ in DEF_ACTIONS and typ != "Pressure" and loc:
            if typ != "Duel" or e["duel"]["type"]["name"] == "Tackle":
                tm["def_actions"] += 1; tm["def_action_x"] += loc[0]
        # PPDA numerator side: tackles, interceptions, fouls in the opponent's own 60% (our x >= 48)
        if loc and loc[0] >= 48 and (typ in ("Interception", "Foul Committed") or (typ == "Duel" and e["duel"]["type"]["name"] == "Tackle")):
            tm["ppda_actions"] += 1
    # possessions per team (distinct possession ids where the team had the ball and made a pass)
    poss = defaultdict(set)
    for e in ev:
        if e["type"]["name"] == "Pass":
            poss[team_name(e["possession_team"]["name"])].add(e["possession"])

    players = []
    for pid, start in on.items():
        mins = off.get(pid, end) - start
        if mins <= 0:
            continue
        pos = pos_counts[pid].most_common(1)[0][0] if pos_counts[pid] else None
        players.append({"match_id": match_id, "player_id": pid, "player": names[pid], "team": team_of[pid],
                        "position": pos, "minutes": mins, **p[pid]})

    team_rows = []
    for team in teams:
        opp = next(x for x in teams if x != team)
        a, b = t[team], t[opp]
        team_rows.append({"match_id": match_id, "team": team, "opponent": opp,
                          **a,
                          "possessions": len(poss[team]),
                          "opp_passes": b["passes"], "opp_passes_own_60": b["passes_own_60"],
                          "xg_against": b["np_xg"], "goals_against": b["np_goals"]})
    return players, team_rows


if __name__ == "__main__":
    ids = matches().match_id.tolist()
    with ProcessPoolExecutor() as pool:
        out = list(pool.map(parse_match, ids, chunksize=8))
    pm = pd.DataFrame([r for ps, _ in out for r in ps])
    tm = pd.DataFrame([r for _, ts in out for r in ts]).fillna(0)
    num = pm.select_dtypes("number").columns
    pm[num] = pm[num].fillna(0)  # stat never happened = 0; position stays None for players who never touched the ball
    m = matches()[["match_id", "league"]]
    pm.merge(m, on="match_id").to_parquet(DATA / "player_match.parquet")
    tm.merge(m, on="match_id").to_parquet(DATA / "team_match.parquet")
    print(f"player-match rows {len(pm):,}  team-match rows {len(tm):,}")
    print(f"minutes per team-match (should be ~11 x ~94): {pm.groupby(['match_id', 'team']).minutes.sum().describe().round(1).to_dict()}")
