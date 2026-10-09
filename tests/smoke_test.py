"""Smoke test: the app loads and every tab renders, on a small synthetic dataset (the real data is private).
Run: python tests/smoke_test.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).parents[1]
DATA = ROOT / "data" / "players_2425.parquet"
sys.path.insert(0, str(ROOT))


def fake_players(n=400, seed=0):
    import app_columns
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"player_id": np.arange(n), "name": [f"Player {i}" for i in range(n)],
                       "age": rng.integers(18, 36, n), "nationality": "Testland", "photo": None,
                       "team": rng.choice(["Club A", "Club B", "Club C", "Club D"], n),
                       "league": rng.choice(["Premier League", "La Liga", "Serie A"], n),
                       "group": rng.choice(["GK", "DEF", "MID", "FWD"], n, p=[.1, .3, .35, .25]),
                       "position": "x", "minutes": rng.integers(300, 3500, n), "ucl_minutes": rng.integers(0, 900, n),
                       "rating": rng.normal(6.9, .3, n), "pass_accuracy": np.nan})
    for c in app_columns.NUMERIC:
        df[c] = rng.gamma(2, 1, n) if not c.endswith("_pct") else rng.uniform(.3, .8, n)
    df.loc[0, "name"] = "Mohamed Salah"
    df.loc[5, "dribble_success_pct"] = np.nan   # missing values must not break similarity
    return df


def main():
    created = not DATA.exists()
    if created:
        DATA.parent.mkdir(exist_ok=True)
        fake_players().to_parquet(DATA)
    try:
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
        assert not at.exception, [e.value for e in at.exception]
        for pid in list(pd.read_parquet(DATA).query("minutes >= 900").player_id[:5]):   # a few players, all positions
            at.selectbox[0].set_value(pid).run()
            assert not at.exception, (pid, [e.value for e in at.exception])
        print("scout smoke test passed")
    finally:
        if created:
            DATA.unlink()


if __name__ == "__main__":
    main()
