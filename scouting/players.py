"""Project 2, steps 2-3 + 5: player profiles, "find similar players", unsupervised role discovery.
Run: .venv/Scripts/python scouting/players.py"""
import sys
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import Radar
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parents[1]))
from statsbomb import DATA

HERE = Path(__file__).parent
MIN_MINUTES = 900

GROUP = {  # StatsBomb position -> broad group
    "Goalkeeper": "GK",
    **dict.fromkeys(["Center Back", "Left Center Back", "Right Center Back"], "CB"),
    **dict.fromkeys(["Left Back", "Right Back", "Left Wing Back", "Right Wing Back"], "FB"),
    **dict.fromkeys(["Center Defensive Midfield", "Left Defensive Midfield", "Right Defensive Midfield",
                     "Center Midfield", "Left Center Midfield", "Right Center Midfield"], "CM"),
    **dict.fromkeys(["Left Midfield", "Right Midfield", "Left Wing", "Right Wing", "Center Attacking Midfield",
                     "Left Attacking Midfield", "Right Attacking Midfield"], "AM/W"),
    **dict.fromkeys(["Center Forward", "Left Center Forward", "Right Center Forward", "Secondary Striker"], "ST"),
}

PER90 = ["np_xg", "shots", "xa", "key_passes", "passes", "prog_passes", "passes_final_third", "passes_into_box",
         "crosses", "through_balls", "long_passes", "switches", "prog_carries", "carries_into_box", "dribbles",
         "tackles", "interceptions", "pressures", "recoveries", "blocks", "clearances", "aerials_won",
         "dispossessed", "miscontrols", "fouls", "actions_in_box"]
# Names read off each cluster's top features at k=8 (random_state=0). Another k -> numbered roles only.
ROLE_NAMES = {0: "Deep-lying playmaker", 1: "Striker", 2: "Box-to-box / pressing mid", 3: "Dribbling winger",
              4: "Centre-back", 5: "Ball-winning mid", 6: "Full-back", 7: "Creative #10"}
# Goals/assists deliberately left out: xG/xA measure the chances, goals add finishing luck.


def player_season():
    pm = pd.read_parquet(DATA / "player_match.parquet")
    pm["group"] = pm.position.map(GROUP)
    stats = pm.drop(columns=["match_id", "player", "team", "position", "group", "league"]).groupby("player_id").sum()
    # identity = the team / position / league where the player spent the most minutes (handles January transfers)
    def main(col):
        return pm.groupby(["player_id", col]).minutes.sum().reset_index().sort_values("minutes").groupby("player_id")[col].last()
    info = pd.DataFrame({c: main(c) for c in ["player", "team", "group", "league"]})
    df = info.join(stats)
    df["player"] = df.player.str.replace("''", "'")  # StatsBomb stores N'Golo as N''Golo
    df = df[(df.minutes >= MIN_MINUTES) & (df.group != "GK")].copy()

    p90 = df.minutes / 90
    feats = pd.DataFrame({f"{c}_p90": df[c] / p90 for c in PER90}, index=df.index)
    safe = lambda a, b: (df[a] / df[b].replace(0, np.nan)).fillna(0)
    feats["pass_cmp_pct"] = safe("passes_cmp", "passes")
    feats["long_cmp_pct"] = safe("long_passes_cmp", "long_passes")
    feats["dribble_success_pct"] = safe("dribbles_won", "dribbles")
    feats["aerial_win_pct"] = (df.aerials_won / (df.aerials_won + df.aerials_lost).replace(0, np.nan)).fillna(0)
    feats["xg_per_shot"] = safe("np_xg", "shots")
    feats["avg_pass_length"] = safe("pass_length", "passes")
    feats["avg_action_x"] = safe("action_x", "actions")            # how high up the pitch they play
    feats["avg_action_width"] = safe("action_width", "actions")    # central vs touchline
    feats["share_att_third"] = safe("actions_att_third", "actions")
    feats["share_def_third"] = safe("actions_def_third", "actions")
    feats["pressures_high_share"] = safe("pressures_att_third", "pressures")
    return df, feats


def norm(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def find(df, name):
    hits = df[df.player.map(norm).str.contains(norm(name))]
    if hits.empty:
        raise KeyError(f"no player matching {name!r} with >= {MIN_MINUTES} minutes")
    return hits.minutes.idxmax()


def similar_players(df, emb, name, n=8, same_group=True):
    """Cosine similarity in standardised + PCA space."""
    pid = find(df, name)
    v = emb.loc[pid]
    sims = emb @ v / (np.linalg.norm(emb, axis=1) * np.linalg.norm(v))
    out = df[["player", "team", "league", "group", "minutes"]].assign(similarity=sims).drop(pid)
    if same_group:
        out = out[out.group == df.loc[pid, "group"]]
    return df.loc[pid, "player"], out.sort_values("similarity", ascending=False).head(n)


if __name__ == "__main__":
    df, feats = player_season()
    print(f"{len(df):,} outfield players with >= {MIN_MINUTES} minutes; groups: {df.group.value_counts().to_dict()}")

    z = pd.DataFrame(StandardScaler().fit_transform(feats), index=feats.index, columns=feats.columns)
    pca = PCA(n_components=0.9, random_state=0).fit(z)   # keep 90% of variance; removes correlated noise (passes vs pass_cmp...)
    emb = pd.DataFrame(pca.transform(z), index=z.index)
    print(f"PCA: {feats.shape[1]} features -> {emb.shape[1]} components for 90% variance")

    pd.set_option("display.width", 200)
    for name in ["Kanté", "Özil", "Mahrez", "Busquets", "Harry Kane", "Dimitri Payet"]:
        who, sim = similar_players(df, emb, name)
        print(f"\nmost similar to {who} ({df.loc[find(df, name), 'team']}):")
        print(sim[["player", "team", "league", "similarity"]].round(3).to_string(index=False))

    # Unsupervised role discovery: cluster ALL outfield players without telling the model their position
    print("\nsilhouette by k:", {k: round(silhouette_score(emb, KMeans(k, n_init=20, random_state=0).fit_predict(emb)), 3) for k in range(3, 13)})
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    km = KMeans(K, n_init=50, random_state=0).fit(emb)
    df["role"] = km.labels_
    names = ROLE_NAMES if K == 8 else {r: str(r) for r in range(K)}
    print(f"\nk={K}: clusters vs labelled position group (model never saw the position):")
    print(pd.crosstab(df.role, df.group))
    centre = z.groupby(df.role).mean()
    for r in range(K):
        top = centre.loc[r].sort_values(ascending=False)
        examples = df[df.role == r].sort_values("minutes", ascending=False).player.head(6).str.split().str[-1].tolist()
        print(f"\nrole {r} = {names[r]} ({(df.role == r).sum()} players) high: {', '.join(top.index[:5])} | low: {', '.join(top.index[-3:])}")
        print(f"  e.g. {', '.join(examples)}")
    df["role_name"] = df.role.map(names)
    df.join(feats).to_parquet(DATA / "player_season.parquet")

    # PCA scatter coloured by discovered role
    fig, ax = plt.subplots(figsize=(9, 7))
    for r in range(K):
        m = df.role == r
        ax.scatter(emb.loc[m, 0], emb.loc[m, 1], s=10, alpha=0.7, color=plt.cm.tab10(r), label=names[r])
    for name, label in [("Kanté", "Kanté"), ("Özil", "Özil"), ("Lionel Andrés Messi", "Messi"), ("Busquets", "Busquets"),
                        ("Harry Kane", "Kane"), ("Sergio Ramos", "Sergio Ramos"), ("Marcelo Vieira", "Marcelo"), ("Mahrez", "Mahrez")]:
        pid = find(df, name)
        ax.annotate(label, (emb.loc[pid, 0], emb.loc[pid, 1]), fontsize=8, weight="bold")
    ax.legend(fontsize=7, loc="best")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    ax.set_title("Outfield players 2015/16, coloured by discovered role")
    fig.savefig(HERE / "player_roles_pca.png", dpi=120, bbox_inches="tight")

    # Radar: Kanté vs his closest match, as percentiles among all outfield players
    a = find(df, "Kanté")
    b = similar_players(df, emb, "Kanté", n=1)[1].index[0]
    params = ["tackles_p90", "interceptions_p90", "pressures_p90", "recoveries_p90", "prog_passes_p90",
              "prog_carries_p90", "pass_cmp_pct", "dribbles_p90", "key_passes_p90", "np_xg_p90"]
    pct = feats[params].rank(pct=True) * 100
    radar = Radar(params, [0] * len(params), [100] * len(params), num_rings=4, ring_width=1, center_circle_radius=1)
    fig, ax = radar.setup_axis()
    radar.draw_circles(ax=ax, facecolor="#eeeeee", edgecolor="#cccccc")
    radar.draw_radar_compare(pct.loc[a], pct.loc[b], ax=ax, kwargs_radar={"facecolor": "#d62728", "alpha": 0.5},
                             kwargs_compare={"facecolor": "#1f77b4", "alpha": 0.5})
    radar.draw_param_labels(ax=ax, fontsize=9)
    ax.set_title(f"{df.loc[a, 'player']} (red) vs {df.loc[b, 'player']} (blue), percentiles", fontsize=10)
    fig.savefig(HERE / "radar_kante.png", dpi=120, bbox_inches="tight")
    print("\nsaved player_roles_pca.png, radar_kante.png, data/player_season.parquet")
