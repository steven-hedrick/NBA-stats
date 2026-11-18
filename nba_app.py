"""
NBA player stats API.

Setup:   pip install fastapi uvicorn nba_api pandas
Run:     uvicorn nba_app:app --reload
Docs:    http://127.0.0.1:8000/docs

Open http://127.0.0.1:8000/ for the web UI (index.html must sit next to this file).

Examples:
  /leaderboard?stat=PTS&limit=25&min_games=5
  /player/Jalen Brunson/home-away
  /player/Jalen Brunson/season
  /player/Jalen Brunson/vs-opponents
  /player/Jalen Brunson/back-to-back
"""
import time

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from nba_api.stats.endpoints import leaguedashplayerstats, playergamelog
from nba_api.stats.static import players

SEASON = "2026-27"
CACHE_SECONDS = 300  # re-fetch from NBA.com at most every 5 minutes

STATS = ["PTS", "REB", "AST", "FG3M", "STL_BLK"]
LABELS = {
    "PTS": "ppg",
    "REB": "rpg",
    "AST": "apg",
    "FG3M": "threes_pg",
    "STL_BLK": "stocks_pg",  # steals + blocks
}

app = FastAPI(title="NBA Player Stats")
_cache: dict[int, tuple[float, pd.DataFrame]] = {}


def find_player(name: str) -> dict:
    matches = players.find_players_by_full_name(name)
    active = [p for p in matches if p["is_active"]]
    pool = active or matches
    if not pool:
        raise HTTPException(404, f"No player found for '{name}'")
    return pool[0]


def load_games(player_id: int) -> pd.DataFrame:
    hit = _cache.get(player_id)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1]

    df = playergamelog.PlayerGameLog(
        player_id=player_id, season=SEASON, timeout=30
    ).get_data_frames()[0]
    if df.empty:
        raise HTTPException(404, f"No games found for {SEASON}")

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    df["OPP"] = df["MATCHUP"].str.split().str[-1]  # "LAL vs. BOS" -> "BOS"
    df["STL_BLK"] = df["STL"] + df["BLK"]
    # Second night of a back-to-back = played the calendar day before
    df["B2B"] = df["GAME_DATE"].diff().dt.days.eq(1)
    df["HOME"] = df["MATCHUP"].str.contains("vs.", regex=False)

    _cache[player_id] = (time.time(), df)
    return df


def averages(df: pd.DataFrame) -> dict:
    out = {"games": int(len(df))}
    for col, label in LABELS.items():
        out[label] = round(float(df[col].mean()), 1) if len(df) else None
    return out


@app.get("/player/{name}/season")
def season(name: str):
    p = find_player(name)
    df = load_games(p["id"])
    return {"player": p["full_name"], "season": SEASON, **averages(df)}


@app.get("/player/{name}/vs-opponents")
def vs_opponents(name: str):
    p = find_player(name)
    df = load_games(p["id"])
    rows = [{"opponent": opp, **averages(g)} for opp, g in df.groupby("OPP")]
    rows.sort(key=lambda r: r["ppg"], reverse=True)
    return {
        "player": p["full_name"],
        "season": SEASON,
        "season_avg": averages(df),
        "by_opponent": rows,
    }


@app.get("/player/{name}/back-to-back")
def back_to_back(name: str):
    p = find_player(name)
    df = load_games(p["id"])
    b2b, rested = averages(df[df["B2B"]]), averages(df[~df["B2B"]])
    diff = {
        label: round(b2b[label] - rested[label], 1)
        for label in LABELS.values()
        if b2b[label] is not None and rested[label] is not None
    }
    return {
        "player": p["full_name"],
        "season": SEASON,
        "second_night_b2b": b2b,
        "other_games": rested,
        "difference": diff,  # negative = worse on a back-to-back
        "note": "Based on games the player appeared in, so a missed first night isn't counted as a back-to-back.",
    }


@app.get("/player/{name}/home-away")
def home_away(name: str):
    p = find_player(name)
    df = load_games(p["id"])
    home, away = averages(df[df["HOME"]]), averages(df[~df["HOME"]])
    diff = {
        label: round(home[label] - away[label], 1)
        for label in LABELS.values()
        if home[label] is not None and away[label] is not None
    }
    return {
        "player": p["full_name"],
        "season": SEASON,
        "home": home,
        "away": away,
        "difference": diff,  # positive = better at home
    }


_lb_cache: dict = {"t": 0.0, "df": None}


@app.get("/leaderboard")
def leaderboard(stat: str = "PTS", limit: int = 25, min_games: int = 5):
    """League-wide per-game leaders. stat: PTS, REB, AST, FG3M, STL_BLK."""
    if stat not in STATS:
        raise HTTPException(400, f"stat must be one of {STATS}")
    if time.time() - _lb_cache["t"] > CACHE_SECONDS or _lb_cache["df"] is None:
        df = leaguedashplayerstats.LeagueDashPlayerStats(
            season=SEASON, per_mode_detailed="PerGame", timeout=30
        ).get_data_frames()[0]
        df["STL_BLK"] = df["STL"] + df["BLK"]
        _lb_cache.update(t=time.time(), df=df)
    df = _lb_cache["df"]
    df = df[df["GP"] >= min_games].sort_values(stat, ascending=False).head(limit)
    return {
        "season": SEASON,
        "stat": stat,
        "leaders": [
            {
                "rank": i + 1,
                "player": r["PLAYER_NAME"],
                "team": r["TEAM_ABBREVIATION"],
                "games": int(r["GP"]),
                **{LABELS[c]: round(float(r[c]), 1) for c in STATS},
            }
            for i, (_, r) in enumerate(df.iterrows())
        ],
    }


@app.get("/players")
def search_players(q: str = ""):
    """Active-player name suggestions for the search box."""
    q = q.lower().strip()
    if len(q) < 2:
        return []
    return [
        p["full_name"]
        for p in players.get_active_players()
        if q in p["full_name"].lower()
    ][:10]


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(Path(__file__).with_name("index.html"))


if __name__ == "__main__":
    # `python nba_app.py` -> reachable from other devices on your Wi-Fi.
    # Hosts like Render/Railway set PORT for you.
    import os

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
