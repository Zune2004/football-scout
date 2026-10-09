"""Scout: find players with a similar statistical profile - 2024/25, top-5 leagues + Champions League.
Method: role-specific per-90 profile, standardised within position, cosine similarity.
Data: API-Football 2024/25 season stats. The data file is private (API-Football forbids redistribution); the deployed
app downloads it with a read-only token from Streamlit secrets. Run: streamlit run app.py"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

import ui

LOCAL = Path(__file__).parent / "data" / "players_2425.parquet"
MIN_DEFAULT = 900
# role-specific profiles; only stats the source fills for nearly every player (pass accuracy is missing for most)
PROFILE = {
    "GK": ["saves_p90", "conceded_p90", "passes_p90", "rating"],
    "DEF": ["tackles_p90", "interceptions_p90", "blocks_p90", "duels_p90", "duel_win_pct", "passes_p90",
            "key_passes_p90", "dribbles_p90", "shots_p90", "fouls_committed_p90"],
    "MID": ["passes_p90", "key_passes_p90", "tackles_p90", "interceptions_p90", "duels_p90", "duel_win_pct",
            "dribbles_p90", "dribble_success_pct", "shots_p90", "goals_p90", "assists_p90"],
    "FWD": ["shots_p90", "shots_on_p90", "goals_p90", "assists_p90", "key_passes_p90", "dribbles_p90",
            "dribble_success_pct", "duels_p90", "duel_win_pct", "fouls_drawn_p90"],
}
HEADLINE = {   # the 4 tiles on the player card
    "GK": ["saves_p90", "conceded_p90", "passes_p90", "rating"],
    "DEF": ["tackles_p90", "interceptions_p90", "duel_win_pct", "key_passes_p90"],
    "MID": ["key_passes_p90", "passes_p90", "tackles_p90", "goals_p90"],
    "FWD": ["goals_p90", "shots_p90", "key_passes_p90", "dribbles_p90"],
}
LOWER_IS_BETTER = {"conceded_p90", "fouls_committed_p90"}
FAMILY = {   # for the "what matters to you" weights
    "Attacking": ["shots_p90", "shots_on_p90", "goals_p90", "assists_p90", "key_passes_p90"],
    "Carrying the ball": ["dribbles_p90", "dribble_success_pct", "fouls_drawn_p90"],
    "Defending": ["tackles_p90", "interceptions_p90", "blocks_p90", "duels_p90", "duel_win_pct", "fouls_committed_p90"],
    "Passing": ["passes_p90"],
    "Goalkeeping": ["saves_p90", "conceded_p90", "rating"],
}
LABELS = {"saves_p90": "Saves /90", "conceded_p90": "Conceded /90", "passes_p90": "Passes /90", "rating": "Rating",
          "tackles_p90": "Tackles /90", "interceptions_p90": "Interceptions /90", "blocks_p90": "Blocks /90",
          "duels_p90": "Duels /90", "duel_win_pct": "Duel win %", "key_passes_p90": "Key passes /90",
          "dribbles_p90": "Dribbles /90", "dribble_success_pct": "Dribble success", "shots_p90": "Shots /90",
          "shots_on_p90": "On target /90", "goals_p90": "Goals /90", "assists_p90": "Assists /90",
          "fouls_drawn_p90": "Fouls won /90", "fouls_committed_p90": "Fouls made /90"}
POS_NAME = {"GK": "Goalkeeper", "DEF": "Defender", "MID": "Midfielder", "FWD": "Forward"}

ui.setup("Scout", "🔎")


@st.cache_data(ttl=6 * 3600)
def load():
    if LOCAL.exists():
        return pd.read_parquet(LOCAL)
    s = st.secrets
    r = requests.get(s["DATA_URL"], headers={"Authorization": f"Bearer {s['GITHUB_TOKEN']}",
                                             "Accept": "application/vnd.github.raw"}, timeout=60)
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


def fmt(col, v):
    if pd.isna(v):
        return "-"
    return f"{v:.0%}" if col.endswith("_pct") else f"{v:.2f}"


def profile_matrix(pool, group):
    cols = PROFILE[group]
    x = pool[pool.group == group][cols]
    x = x.fillna(x.median()).fillna(0)          # e.g. no dribbles -> no success rate: use the position's typical value
    return (x - x.mean()) / x.std().replace(0, 1).fillna(1)


def similar(pool, pid, keep, weights=None):
    z = profile_matrix(pool, pool.at[pid, "group"])
    if weights:   # scale each family's columns: 0 = ignore, 2 = counts double
        for fam, w in weights.items():
            cols = [c for c in FAMILY[fam] if c in z.columns]
            z[cols] = z[cols] * w
    v = z.loc[pid]
    sim = ((z @ v) / (np.linalg.norm(z, axis=1) * np.linalg.norm(v) + 1e-9)).fillna(0)
    out = pool.loc[z.index].assign(similarity=sim).drop(pid)
    return out[keep(out)].sort_values("similarity", ascending=False)


def percentiles(pool, group):
    cols = PROFILE[group]
    g = pool[pool.group == group][cols]
    pct = g.rank(pct=True) * 100
    for c in LOWER_IS_BETTER & set(cols):
        pct[c] = 100 - pct[c] + 100 / len(g)
    return pct


df = load().set_index("player_id")
df = df[df.group.notna()]

with st.sidebar:
    st.markdown("### Filters")
    min_minutes = st.slider("Minimum minutes played", 300, 3000, MIN_DEFAULT, 100,
                            help="Per-90 numbers from a handful of games are mostly noise.")
    leagues_all = sorted(df.league.dropna().unique())
    leagues = st.multiselect("Show matches from", leagues_all, default=leagues_all)
    max_age = st.slider("Maximum age (for suggestions)", 17, 40, 40)
    st.markdown("<div class='note'>Data: API-Football, 2024/25. Top-5 leagues + Champions League.</div>",
                unsafe_allow_html=True)
pool = df[df.minutes >= min_minutes]

ui.hero("Scout", "Find players with the same statistical profile, anywhere in Europe's top five leagues. "
                 "Pick a player and get his closest matches, younger or cheaper alternatives, and how he compares.",
        tag="2024/25 season · Top-5 leagues + Champions League")
ui.kpis([("Players", f"{len(pool):,}", f"with {min_minutes}+ minutes", "green"),
         ("Clubs", f"{pool.team.nunique()}", "across 5 leagues", "blue"),
         ("In the Champions League", f"{(pool.ucl_minutes > 0).sum():,}", "players with UCL minutes", "amber"),
         ("Method", "Cosine", "on role-specific per-90 profiles", "")])

tab_sim, tab_cmp, tab_board, tab_how = st.tabs(["Find similar players", "Compare two players", "Leaderboards",
                                                "How it works"])

with tab_sim:
    order = pool.sort_values("minutes", ascending=False)
    label = (order.name + " · " + order.team).to_dict()
    default = order.index[order.name.eq("Mohamed Salah")]
    pick = st.selectbox("Search a player", order.index, index=int(order.index.get_loc(default[0])) if len(default) else 0,
                        format_func=lambda i: label[i])
    me = pool.loc[pick]
    pct = percentiles(pool, me.group)
    tiles = [(LABELS[c], fmt(c, me[c]), pct.at[pick, c]) for c in HEADLINE[me.group]]
    pills = "".join(f"<span class='pill'>{ui.esc(p)}</span>" for p in
                    [POS_NAME[me.group], f"Age {int(me.age)}" if pd.notna(me.age) else None,
                     f"{int(me.minutes):,} min", f"{int(me.ucl_minutes):,} UCL min" if me.ucl_minutes else None,
                     f"Rating {me.rating:.2f}" if pd.notna(me.rating) else None] if p)
    st.markdown(f"<div class='card pcard'><div class='name'>{ui.esc(me['name'])}</div>"
                f"<div class='meta'>{ui.esc(me.team)} · {ui.esc(me.league)} · {ui.esc(me.nationality)}</div>{pills}"
                f"{ui.stat_tiles(tiles)}</div>", unsafe_allow_html=True)

    st.markdown("### Most similar players")
    fams = [f for f in FAMILY if set(FAMILY[f]) & set(PROFILE[me.group])]
    with st.expander("What matters most to you? Weight the similarity"):
        cols_w = st.columns(len(fams))
        weights = {f: c.select_slider(f, options=[0.0, 0.5, 1.0, 1.5, 2.0], value=1.0,
                                      format_func=lambda v: {0.0: "ignore", 0.5: "less", 1.0: "normal", 1.5: "more", 2.0: "double"}[v])
                   for f, c in zip(fams, cols_w)}
    keep = lambda d: d.league.isin(leagues) & (d.age <= max_age)
    sims = similar(pool, pick, keep, weights)
    top = sims.head(9)
    if top.empty:
        st.info("Nobody matches these filters. Widen the leagues or the age limit.")
    else:
        cards = []
        for i, r in top.iterrows():
            k1, k2 = HEADLINE[me.group][:2]
            cards.append(f"<div class='sim'><span class='pct'>{r.similarity:.0%}</span><div class='n'>{ui.esc(r['name'])}</div>"
                         f"<div class='c'>{ui.esc(r.team)} · age {int(r.age) if pd.notna(r.age) else '-'}</div>"
                         f"<div class='bar'><span style='width:{max(r.similarity, 0) * 100:.0f}%;background:{ui.GREEN}'></span></div>"
                         f"<div class='c' style='margin-top:6px'>{LABELS[k1]} {fmt(k1, r[k1])} · {LABELS[k2]} {fmt(k2, r[k2])}</div></div>")
        st.markdown(f"<div class='simgrid'>{''.join(cards)}</div>", unsafe_allow_html=True)

        st.markdown("### Side by side")
        other = st.selectbox("Compare with", top.index, format_func=lambda i: label.get(i, df.at[i, "name"]))
        cols = PROFILE[me.group]
        theta = [LABELS[c] for c in cols] + [LABELS[cols[0]]]
        fig = go.Figure()
        for i, colour in ((pick, ui.GREEN), (other, ui.AMBER)):
            r = pct.loc[i, cols].tolist()
            fig.add_trace(go.Scatterpolar(r=r + r[:1], theta=theta, fill="toself", name=df.at[i, "name"],
                                          line=dict(color=colour, width=2), opacity=0.55,
                                          hovertemplate="%{theta}: %{r:.0f}th percentile<extra>" + df.at[i, "name"] + "</extra>"))
        fig.update_layout(height=520, margin=dict(l=60, r=60, t=30, b=30), legend=dict(orientation="h", y=-0.08),
                          polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 100], gridcolor=ui.LINE,
                                     tickfont=dict(color=ui.MUTED, size=10)), angularaxis=dict(gridcolor=ui.LINE)))
        st.plotly_chart(fig, width="stretch")
        st.caption("Percentiles among players in the same position with the minimum minutes. "
                   "For 'conceded' and 'fouls made', higher percentile = fewer.")

with tab_cmp:
    allp = pool.sort_values("minutes", ascending=False)
    lab = (allp.name + " · " + allp.team).to_dict()
    c1, c2 = st.columns(2)
    a = c1.selectbox("Player A", allp.index, index=int(allp.index.get_loc(pick)), format_func=lambda i: lab[i], key="cmp_a")
    same = allp[allp.group == allp.at[a, "group"]]
    b = c2.selectbox("Player B (same position)", [i for i in same.index if i != a], format_func=lambda i: lab[i], key="cmp_b")
    g = allp.at[a, "group"]
    cols = PROFILE[g]
    pct_c = percentiles(pool, g)
    theta = [LABELS[c] for c in cols] + [LABELS[cols[0]]]
    fig = go.Figure()
    for i, colour in ((a, ui.GREEN), (b, ui.AMBER)):
        r = pct_c.loc[i, cols].tolist()
        fig.add_trace(go.Scatterpolar(r=r + r[:1], theta=theta, fill="toself", name=allp.at[i, "name"],
                                      line=dict(color=colour, width=2), opacity=0.55))
    fig.update_layout(height=520, margin=dict(l=60, r=60, t=30, b=30), legend=dict(orientation="h", y=-0.08),
                      polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 100], gridcolor=ui.LINE,
                                 tickfont=dict(color=ui.MUTED, size=10)), angularaxis=dict(gridcolor=ui.LINE)))
    st.plotly_chart(fig, width="stretch")
    rows = []
    for c in cols:
        va, vb = allp.at[a, c], allp.at[b, c]
        better = "" if pd.isna(va) or pd.isna(vb) or va == vb else (
            allp.at[a, "name"] if (va < vb) == (c in LOWER_IS_BETTER) else allp.at[b, "name"])
        rows.append({"stat": LABELS[c], allp.at[a, "name"]: fmt(c, va), allp.at[b, "name"]: fmt(c, vb), "better": better})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

with tab_board:
    c1, c2 = st.columns(2)
    group = c1.segmented_control("Position", ["FWD", "MID", "DEF", "GK"], default="FWD",
                                 format_func=lambda g: POS_NAME[g]) or "FWD"
    stat = c2.selectbox("Stat", PROFILE[group], format_func=lambda c: LABELS[c])
    board = pool[(pool.group == group) & pool.league.isin(leagues) & (pool.age <= max_age)].dropna(subset=[stat])
    board = board.sort_values(stat, ascending=stat in LOWER_IS_BETTER).head(15)
    fig = go.Figure(go.Bar(x=board[stat], y=board.name + "  ·  " + board.team, orientation="h",
                           marker=dict(color=ui.GREEN), text=[fmt(stat, v) for v in board[stat]], textposition="outside",
                           hovertemplate="%{y}<br>" + LABELS[stat] + ": %{x:.2f}<extra></extra>"))
    fig.update_layout(height=40 * len(board) + 80, margin=dict(l=10, r=40, t=10, b=10),
                      yaxis=dict(autorange="reversed"), xaxis=dict(title=LABELS[stat]))
    st.plotly_chart(fig, width="stretch")

with tab_how:
    st.markdown("""
#### How "similar" is measured
1. **Profile:** each player is described by the per-90-minute stats that matter for his position
   (a centre-back is about tackles, interceptions and duels; a forward about shots, goals, dribbles).
2. **Standardise:** every stat is converted to "how far above or below the typical player in that position",
   so passes (dozens per game) don't drown out goals (fractions per game).
3. **Compare shapes:** cosine similarity compares the *shape* of two profiles, not the volume. A player with
   the same mix of actions scores high even if he played fewer minutes.

#### Good to know
- 2024/25 is the newest season available free (API-Football's free plan covers 2022-2024).
- Transferred players' seasons are shown once, at the club they actually played for.
- The 2015/16 deep dive (event data, unsupervised roles, team styles) lives in this repository's `scouting/` folder.
""")
