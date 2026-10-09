"""Project 2, step 4: cluster the 78 teams by HOW they play (not how well).
Run: .venv/Scripts/python scouting/teams.py [k]"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parents[1]))
from statsbomb import DATA, matches

HERE = Path(__file__).parent
# Read off the cluster centres at k=4 (random_state=0). Another k -> numbered clusters only.
STYLE_NAMES = {0: "Passive low block", 1: "Possession & high press", 2: "Aggressive press, wide",
               3: "Direct & counter"}


def team_season():
    tm = pd.read_parquet(DATA / "team_match.parquet")
    s = tm.drop(columns=["match_id", "opponent"]).groupby(["team", "league"]).sum()
    s["games"] = tm.groupby(["team", "league"]).size()
    style = pd.DataFrame({
        "possession": s.passes / (s.passes + s.opp_passes),               # pass share as possession proxy
        "pass_cmp_pct": s.passes_cmp / s.passes,
        "passes_per_possession": s.passes / s.possessions,                # patient build-up vs quick attacks
        "avg_pass_length": s.pass_length / s.passes,
        "long_ball_share": s.long_passes / s.passes,
        "ppda": s.opp_passes_own_60 / s.ppda_actions,                      # LOW = intense press
        "high_press_share": s.pressures_att_third / s.pressures,          # where the pressing happens
        "def_line_height": s.def_action_x / s.def_actions,                # avg x of defensive actions
        "crosses_per_entry": s.crosses / s.final_third_entries,           # wide & crossing vs combining
        "central_entry_share": s.central_entries / s.final_third_entries,
        "counter_shot_share": s.counter_shots / s.shots,
    })
    # Outcomes: NOT used for clustering, only to see whether a style goes with success
    m = matches()
    pts = pd.concat([
        m.assign(team=m.home, pts=np.select([m.home_score > m.away_score, m.home_score == m.away_score], [3, 1], 0)),
        m.assign(team=m.away, pts=np.select([m.away_score > m.home_score, m.home_score == m.away_score], [3, 1], 0)),
    ]).groupby("team").pts.sum()
    outcome = pd.DataFrame({
        "points": pts.reindex(s.index.get_level_values("team")).to_numpy(),
        "np_xg_diff_pg": ((s.np_xg - s.xg_against) / s.games).to_numpy(),
    }, index=s.index)
    return style, outcome


if __name__ == "__main__":
    style, outcome = team_season()
    print(f"{len(style)} teams")
    z = pd.DataFrame(StandardScaler().fit_transform(style), index=style.index, columns=style.columns)

    print("silhouette by k:", {k: round(silhouette_score(z, KMeans(k, n_init=50, random_state=0).fit_predict(z)), 3) for k in range(2, 9)})
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    style["cluster"] = KMeans(K, n_init=100, random_state=0).fit_predict(z)
    names = STYLE_NAMES if K == 4 else {c: str(c) for c in range(K)}
    style["style"] = style.cluster.map(names)

    pd.set_option("display.width", 220)
    print("\ncluster centres (z-scores; + = above average):")
    print(z.groupby(style["style"]).mean().round(2).T.to_string())
    full = style.join(outcome)
    print("\nstyle outcomes:")
    print(full.groupby("style").agg(teams=("points", "size"), avg_points=("points", "mean"), xg_diff_pg=("np_xg_diff_pg", "mean")).round(2).to_string())
    for c in range(K):
        members = full[full.cluster == c].sort_values("points", ascending=False)
        print(f"\n{names[c]}: " + ", ".join(f"{t} ({lg.split()[0]}, {p})" for (t, lg), p in members.points.items()))
    full.to_parquet(DATA / "team_season.parquet")

    # Style map on the two most readable axes: how much ball, how hard you press
    fig, ax = plt.subplots(figsize=(10, 7))
    for c in range(K):
        d = full[full.cluster == c]
        ax.scatter(d.possession * 100, d.ppda, s=d.points * 1.5, alpha=0.6, color=plt.cm.tab10(c), label=names[c])
    for team in ["Barcelona", "Leicester City", "Atlético Madrid", "Juventus", "Paris Saint-Germain", "Stoke City",
                 "Arsenal", "Tottenham Hotspur", "Napoli", "West Ham United", "Real Madrid", "Southampton"]:
        r = full.xs(team, level="team").iloc[0]
        ax.annotate(team, (r.possession * 100, r.ppda), fontsize=8)
    ax.invert_yaxis()  # low PPDA = intense press -> top
    ax.set_xlabel("possession % (pass share)")
    ax.set_ylabel("PPDA (opponent passes per defensive action, lower = more pressing)")
    ax.set_title("Team playing styles 2015/16, top-4 leagues (bubble size = points)")
    ax.legend(fontsize=8)
    fig.savefig(HERE / "team_styles.png", dpi=120, bbox_inches="tight")
    print("\nsaved team_styles.png, data/team_season.parquet")
