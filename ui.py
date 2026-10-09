"""Shared look for the Football Labs apps: dark 'floodlit pitch' theme, KPI cards, player cards, plotly template."""
import html

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

BG, CARD, LINE, TEXT, MUTED = "#0B1220", "#111A2E", "#1F2A44", "#E5E9F0", "#94A3B8"
GREEN, AMBER, ROSE, BLUE = "#22C55E", "#F59E0B", "#F43F5E", "#60A5FA"

pio.templates["labs"] = go.layout.Template(layout=dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Fira Sans, sans-serif", color=TEXT, size=13),
    colorway=[GREEN, AMBER, BLUE, ROSE, "#A78BFA", "#2DD4BF"],
    xaxis=dict(gridcolor=LINE, zerolinecolor=LINE), yaxis=dict(gridcolor=LINE, zerolinecolor=LINE),
    hoverlabel=dict(bgcolor=CARD, bordercolor=LINE, font=dict(color=TEXT)),
    legend=dict(bgcolor="rgba(0,0,0,0)")))
pio.templates.default = "labs"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@500;600&family=Fira+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Fira Sans', sans-serif; }}
.block-container {{ padding-top: 2rem; max-width: 1200px; }}
.hero h1 {{ font-size: 2.4rem; font-weight: 700; margin: 0; letter-spacing: -0.02em; }}
.hero p {{ color: {MUTED}; font-size: 1.02rem; margin: .35rem 0 0 0; line-height: 1.5; }}
.hero .tag {{ display:inline-block; background: rgba(34,197,94,.12); color:{GREEN}; border:1px solid rgba(34,197,94,.35);
  border-radius: 999px; padding: 2px 10px; font-size: .78rem; font-weight:600; margin-bottom:.6rem; }}
.kpis {{ display:flex; flex-wrap:wrap; gap:12px; margin: 1rem 0 0.5rem 0; }}
.kpi {{ flex: 1 1 180px; background:{CARD}; border:1px solid {LINE}; border-radius:14px; padding:14px 16px; }}
.kpi .label {{ color:{MUTED}; font-size:.78rem; text-transform:uppercase; letter-spacing:.06em; font-weight:600; }}
.kpi .value {{ font-family:'Fira Code', monospace; font-size:1.55rem; font-weight:600; margin-top:4px; }}
.kpi .sub {{ color:{MUTED}; font-size:.85rem; margin-top:2px; line-height:1.35; }}
.kpi .value.word {{ font-family:'Fira Sans', sans-serif; font-size:1.25rem; font-weight:700; line-height:1.25; }}
.kpi.green .value {{ color:{GREEN}; }} .kpi.amber .value {{ color:{AMBER}; }}
.kpi.rose .value {{ color:{ROSE}; }} .kpi.blue .value {{ color:{BLUE}; }}
.card {{ background:{CARD}; border:1px solid {LINE}; border-radius:16px; padding:18px 20px; }}
.pcard .name {{ font-size:1.6rem; font-weight:700; letter-spacing:-.01em; }}
.pcard .meta {{ color:{MUTED}; margin-top:2px; }}
.pill {{ display:inline-block; border-radius:999px; padding:3px 10px; font-size:.8rem; font-weight:600; margin:8px 6px 0 0;
  background:rgba(96,165,250,.12); color:{BLUE}; border:1px solid rgba(96,165,250,.3); }}
.stats {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap:10px; margin-top:14px; }}
.stat {{ background:{BG}; border:1px solid {LINE}; border-radius:12px; padding:10px 12px; }}
.stat .k {{ color:{MUTED}; font-size:.75rem; font-weight:600; text-transform:uppercase; letter-spacing:.05em; }}
.stat .v {{ font-family:'Fira Code', monospace; font-size:1.2rem; font-weight:600; }}
.bar {{ height:6px; background:{LINE}; border-radius:6px; margin-top:6px; overflow:hidden; }}
.bar > span {{ display:block; height:100%; border-radius:6px; }}
.simgrid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap:12px; }}
.sim {{ background:{CARD}; border:1px solid {LINE}; border-radius:14px; padding:12px 14px; }}
.sim .n {{ font-weight:700; }} .sim .c {{ color:{MUTED}; font-size:.85rem; }}
.sim .pct {{ font-family:'Fira Code', monospace; color:{GREEN}; font-weight:600; float:right; }}
.note {{ color:{MUTED}; font-size:.85rem; line-height:1.5; }}
h2, h3 {{ letter-spacing:-.01em; }}
@media (max-width: 640px) {{
  .hero h1 {{ font-size: 1.9rem; }}
  .kpi {{ flex: 1 1 40%; padding: 10px 12px; }}
  .kpi .value {{ font-size: 1.2rem; }}
  .kpi .sub {{ font-size: .78rem; }}
  .pcard .name {{ font-size: 1.3rem; }}
}}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
</style>
"""


def setup(title, icon):
    st.set_page_config(page_title=title, page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)


def esc(x):
    return html.escape(str(x))


def hero(title, subtitle, tag=None):
    st.markdown(f"<div class='hero'>{f'<span class=tag>{esc(tag)}</span>' if tag else ''}<h1>{esc(title)}</h1>"
                f"<p>{esc(subtitle)}</p></div>", unsafe_allow_html=True)


def kpis(items):
    """items: [(label, value, sub, tone)] with tone in green/amber/rose/blue/''."""
    cards = "".join(f"<div class='kpi {tone}'><div class='label'>{esc(l)}</div>"
                    f"<div class='value{' word' if any(c.isalpha() for c in str(v)) and len(str(v)) > 8 else ''}'>{esc(v)}</div>"
                    f"<div class='sub'>{esc(s)}</div></div>" for l, v, s, tone in items)
    st.markdown(f"<div class='kpis'>{cards}</div>", unsafe_allow_html=True)


def pct_colour(p):
    return GREEN if p >= 80 else BLUE if p >= 50 else AMBER if p >= 25 else ROSE


def stat_tiles(items):
    """items: [(label, value_text, percentile 0-100 or None)]"""
    tiles = []
    for k, v, p in items:
        bar = (f"<div class='bar'><span style='width:{p:.0f}%;background:{pct_colour(p)}'></span></div>"
               f"<div class='note'>{p:.0f}th percentile</div>") if p is not None else ""
        tiles.append(f"<div class='stat'><div class='k'>{esc(k)}</div><div class='v'>{esc(v)}</div>{bar}</div>")
    return f"<div class='stats'>{''.join(tiles)}</div>"
