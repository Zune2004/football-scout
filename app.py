"""Scout: find players with a similar statistical profile - 2024/25, top-5 leagues + Champions League.
Method: per-90 profile by position, standardised, cosine similarity (as in the 2015/16 scouting project here).
Data: API-Football (2024/25 player season stats). The data file is private (API-Football forbids redistribution);
the deployed app downloads it with a read-only token from Streamlit secrets. Run: streamlit run app.py"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

LOCAL = Path(__file__).parent / "data" / "players_2425.parquet"
PROFILE = {   # what each role is about, per 90 minutes
    "GK": ["saves_p90", "conceded_p90", "passes_p90", "pass_accuracy", "rating"],
    "DEF": ["tackles_p90", "interceptions_p90", "blocks_p90", "duels_p90", "duel_win_pct", "passes_p90", "pass_accuracy",
            "key_passes_p90", "dribbles_p90", "shots_p90", "fouls_committed_p90"],
    "MID": ["passes_p90", "pass_accuracy", "key_passes_p90", "tackles_p90", "interceptions_p90", "duels_p90",
            "duel_win_pct", "dribbles_p90", "dribble_success_pct", "shots_p90", "goals_p90", "assists_p90"],
    "FWD": ["shots_p90", "shots_on_p90", "goals_p90", "assists_p90", "key_passes_p90", "dribbles_p90",
            "dribble_success_pct", "duels_p90", "duel_win_pct", "fouls_drawn_p90", "passes_p90"],
}
LABELS = {"saves_p90": "Saves", "conceded_p90": "Conceded", "passes_p90": "Passes", "pass_accuracy": "Pass acc.",
          "rating": "Rating", "tackles_p90": "Tackles", "interceptions_p90": "Interceptions", "blocks_p90": "Blocks",
          "duels_p90": "Duels", "duel_win_pct": "Duel win %", "key_passes_p90": "Key passes", "dribbles_p90": "Dribbles",
          "dribble_success_pct": "Dribble success", "shots_p90": "Shots", "shots_on_p90": "Shots on target",
          "goals_p90": "Goals", "assists_p90": "Assists", "fouls_drawn_p90": "Fouls drawn",
          "fouls_committed_p90": "Fouls committed"}

st.set_page_config(page_title="Scout", page_icon="🔎", layout="wide")


@st.cache_data(ttl=6 * 3600)
def load():
    if LOCAL.exists():
        return pd.read_parquet(LOCAL)
    s = st.secrets
    r = requests.get(s["DATA_URL"], headers={"Authorization": f"Bearer {s['GITHUB_TOKEN']}",
                                             "Accept": "application/vnd.github.raw"}, timeout=60)
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


def similar(df, pid, n, filt):
    me = df.loc[pid]
    cols = PROFILE[me.group]
    pool = df[df.group == me.group]
    x = pool[cols].fillna(pool[cols].median())   # e.g. no dribbles -> no success rate: use the position's typical value
    z = (x - x.mean()) / x.std().replace(0, 1)
    v = z.loc[pid]
    sim = ((z @ v) / (np.linalg.norm(z, axis=1) * np.linalg.norm(v) + 1e-9)).fillna(0)
    out = pool.assign(similarity=sim).drop(pid)
    return out[filt(out)].sort_values("similarity", ascending=False).head(n)


def radar(df, ids, group):
    cols = PROFILE[group]
    pct = df[df.group == group][cols].rank(pct=True) * 100
    fig = go.Figure()
    for i, colour in zip(ids, ["#37003c", "#00b8a9"]):
        r = pct.loc[i].tolist()
        fig.add_trace(go.Scatterpolar(r=r + r[:1], theta=[LABELS[c] for c in cols] + [LABELS[cols[0]]], fill="toself",
                                      name=df.at[i, "name"], line=dict(color=colour), opacity=0.6))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], ticksuffix="%")), height=480,
                      margin=dict(l=40, r=40, t=40, b=40), legend=dict(orientation="h"))
    return fig


df = load().set_index("player_id")
st.title("Scout")
st.caption("Find players with a similar statistical profile. 2024/25 season, top-5 leagues + Champions League "
           "(the newest season available free). Percentiles compare players in the same position.")

with st.sidebar:
    st.header("Filters")
    min_minutes = st.slider("Minimum minutes", 300, 3000, 900, 100)
    leagues = st.multiselect("Leagues", sorted(df.league.dropna().unique()), default=sorted(df.league.dropna().unique()))
    max_age = st.slider("Maximum age", 17, 40, 40)
    st.caption(f"{(df.minutes >= min_minutes).sum():,} players with {min_minutes}+ minutes")
pool = df[(df.minutes >= min_minutes) & df.group.notna()]

tab_sim, tab_board = st.tabs(["Find similar players", "Leaderboards"])
with tab_sim:
    options = pool.sort_values("minutes", ascending=False)
    label = options.name + " (" + options.team + ")"
    pick = st.selectbox("Player", options.index, format_func=lambda i: label[i])
    filt = lambda d: d.league.isin(leagues) & (d.age <= max_age)
    top = similar(pool, pick, 10, filt)
    me = pool.loc[pick]
    st.markdown(f"**{me['name']}**, {me.team} ({me.league}), {me.group}, age {me.age}: {int(me.minutes)} minutes"
                + (f", {int(me.ucl_minutes)} in the Champions League" if me.ucl_minutes else "")
                + (f", average rating {me.rating:.2f}" if pd.notna(me.rating) else ""))
    show = top[["name", "team", "league", "age", "minutes", "rating", "similarity"]].copy()
    show["similarity"] = show.similarity.map(lambda v: f"{v:.0%}")
    st.dataframe(show.round(2), width="stretch", hide_index=True)
    if len(top):
        other = st.selectbox("Compare on the radar with", top.index, format_func=lambda i: label.get(i, df.at[i, "name"]))
        st.plotly_chart(radar(pool, [pick, other], me.group), width="stretch")

with tab_board:
    c1, c2 = st.columns(2)
    group = c1.selectbox("Position", ["FWD", "MID", "DEF", "GK"])
    stat = c2.selectbox("Stat", PROFILE[group], format_func=lambda c: LABELS[c])
    board = pool[(pool.group == group) & pool.league.isin(leagues) & (pool.age <= max_age)]
    board = board.sort_values(stat, ascending=stat in ("conceded_p90", "fouls_committed_p90")).head(25)
    st.dataframe(board[["name", "team", "league", "age", "minutes", stat]].round(2), width="stretch", hide_index=True)

st.divider()
st.caption("Data: API-Football (2024/25). Method: standardised per-90 profiles and cosine similarity within position.")
