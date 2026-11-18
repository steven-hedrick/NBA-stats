"""
NBA player stats API.

Setup:   pip install fastapi uvicorn nba_api pandas requests
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

import os
import unicodedata
from pathlib import Path

import pandas as pd
import requests
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

# Workaround for NBA.com blocking hosting providers:
#  - browser-like headers on every request
#  - optional PROXY_URL env var (e.g. http://user:pass@host:port) to route
#    requests through a proxy with a non-datacenter IP
#  - retries, and stale cached data if NBA.com is unreachable
PROXY = os.environ.get("PROXY_URL") or None
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}


def nba_fetch(endpoint, retries: int = 1, **params):
    """Call an nba_api endpoint with headers/proxy and simple retry."""
    last = None
    for attempt in range(retries):
        try:
            return endpoint(
                headers=HEADERS, proxy=PROXY, timeout=10, **params
            ).get_data_frames()[0]
        except Exception as e:  # network errors, timeouts, blocks
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise HTTPException(
        503,
        "Couldn't reach NBA.com (it may be blocking this server). "
        f"Set PROXY_URL or run from a home network. Detail: {type(last).__name__}",
    )


app = FastAPI(title="NBA Player Stats")
_cache: dict[int, tuple[float, pd.DataFrame]] = {}


def find_player(name: str) -> dict:
    matches = players.find_players_by_full_name(name)
    active = [p for p in matches if p["is_active"]]
    pool = active or matches
    if not pool:
        raise HTTPException(404, f"No player found for '{name}'")
    return pool[0]


ESPN_SEARCH = "https://site.web.api.espn.com/apis/common/v3/search"
ESPN_GAMELOG = (
    "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/"
    "athletes/{id}/gamelog"
)


def espn_get(url: str, **params) -> dict:
    r = requests.get(
        url,
        params=params,
        headers=HEADERS,
        timeout=20,
        proxies={"http": PROXY, "https": PROXY} if PROXY else None,
    )
    r.raise_for_status()
    return r.json()


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def _find_nba_athletes(node, out):
    """Walk any JSON shape and collect NBA athlete uids like s:40~l:46~a:3945274."""
    if isinstance(node, dict):
        uid = node.get("uid")
        if isinstance(uid, str) and "~l:46~a:" in uid:
            out.append(uid.split("~a:")[-1])
        for v in node.values():
            _find_nba_athletes(v, out)
    elif isinstance(node, list):
        for v in node:
            _find_nba_athletes(v, out)


def espn_athlete_id(name: str) -> str:
    for query in (_strip_accents(name), name):
        data = espn_get(ESPN_SEARCH, query=query, limit=10, mode="prefix")
        ids: list = []
        _find_nba_athletes(data, ids)
        if ids:
            return ids[0]
    raise HTTPException(404, f"ESPN has no NBA player named '{name}'")


@app.get("/debug/espn-search/{name}")
def debug_espn_search(name: str):
    """Troubleshooting: raw ESPN search response."""
    return espn_get(ESPN_SEARCH, query=_strip_accents(name), limit=5, mode="prefix")


def espn_games(name: str) -> pd.DataFrame:
    """Free fallback: ESPN's unofficial game log for the current season."""
    data = espn_get(ESPN_GAMELOG.format(id=espn_athlete_id(name)))
    labels = data.get("labels") or data.get("names") or []
    idx = {label: i for i, label in enumerate(labels)}
    events = data.get("events", {})
    rows = []
    for st in data.get("seasonTypes", []):
        if "regular" not in st.get("displayName", "").lower():
            continue
        for cat in st.get("categories", []):
            for ev in cat.get("events", []):
                meta = events.get(str(ev.get("eventId")))
                stats = ev.get("stats", [])
                if not meta or len(stats) < len(labels):
                    continue
                try:
                    rows.append(
                        {
                            "GAME_DATE": meta["gameDate"],
                            "OPP": meta["opponent"]["abbreviation"],
                            "HOME": meta.get("atVs") == "vs",
                            "PTS": float(stats[idx["PTS"]]),
                            "REB": float(stats[idx["REB"]]),
                            "AST": float(stats[idx["AST"]]),
                            "FG3M": float(str(stats[idx["3PT"]]).split("-")[0]),
                            "STL": float(stats[idx["STL"]]),
                            "BLK": float(stats[idx["BLK"]]),
                        }
                    )
                except (KeyError, ValueError, TypeError):
                    continue  # skip rows we can't parse (e.g. DNP)
    if not rows:
        raise HTTPException(404, "ESPN returned no parsable games for this season")
    df = pd.DataFrame(rows)
    # ESPN dates are UTC; convert to US Eastern so late games land on the right day
    df["GAME_DATE"] = (
        pd.to_datetime(df["GAME_DATE"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    return df


def load_games(p: dict) -> pd.DataFrame:
    player_id = p["id"]
    hit = _cache.get(player_id)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1]

    try:
        df = nba_fetch(
            playergamelog.PlayerGameLog, player_id=player_id, season=SEASON
        )
        if df.empty:
            raise HTTPException(404, f"No games found for {SEASON}")
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df["OPP"] = df["MATCHUP"].str.split().str[-1]  # "LAL vs. BOS" -> "BOS"
        df["HOME"] = df["MATCHUP"].str.contains("vs.", regex=False)
    except HTTPException as nba_err:
        if nba_err.status_code == 404:
            raise
        try:  # NBA.com blocked us -> try ESPN
            df = espn_games(p["full_name"])
        except HTTPException:
            raise
        except Exception as e:
            if hit:  # serve stale data rather than failing
                return hit[1]
            raise HTTPException(
                503, f"NBA.com and ESPN both failed ({type(e).__name__})"
            )

    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    df["STL_BLK"] = df["STL"] + df["BLK"]
    # Second night of a back-to-back = played the calendar day before
    df["B2B"] = df["GAME_DATE"].diff().dt.days.eq(1)

    _cache[player_id] = (time.time(), df)
    return df


@app.get("/debug/espn/{name}")
def debug_espn(name: str):
    """Troubleshooting: shows the raw shape of ESPN's response."""
    data = espn_get(ESPN_GAMELOG.format(id=espn_athlete_id(name)))
    return {
        "top_level_keys": list(data.keys()),
        "labels": data.get("labels"),
        "season_types": [st.get("displayName") for st in data.get("seasonTypes", [])],
        "sample_event": next(iter(data.get("events", {}).values()), None),
    }


def averages(df: pd.DataFrame) -> dict:
    out = {"games": int(len(df))}
    for col, label in LABELS.items():
        out[label] = round(float(df[col].mean()), 1) if len(df) else None
    return out


@app.get("/player/{name}/season")
def season(name: str):
    p = find_player(name)
    df = load_games(p)
    return {"player": p["full_name"], "season": SEASON, **averages(df)}


@app.get("/player/{name}/vs-opponents")
def vs_opponents(name: str):
    p = find_player(name)
    df = load_games(p)
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
    df = load_games(p)
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
    df = load_games(p)
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
        try:
            df = nba_fetch(
                leaguedashplayerstats.LeagueDashPlayerStats,
                season=SEASON,
                per_mode_detailed="PerGame",
            )
        except HTTPException:
            if _lb_cache["df"] is None:
                raise
            df = _lb_cache["df"]
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
