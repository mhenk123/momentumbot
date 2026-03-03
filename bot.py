#!/usr/bin/env python3
"""
Hyperliquid Momentum Paper-Trading Bot
=======================================
Fetches live market data from Hyperliquid, simulates trades locally
with a virtual portfolio, applies RSI/momentum filters, two-stage
risk management (hard stop + trailing stop), and self-optimises via
weekly ML (RandomForest). Sends Telegram notifications.

No wallet connection required – all trades are paper-only.
"""

import os
import json
import time
import logging
import sqlite3
import requests
import schedule
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from dotenv import load_dotenv
import pytz
from sklearn.ensemble import RandomForestRegressor

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

CONFIG_PATH = "data/config.json"
DB_PATH     = "data/database.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """Load config.json from disk. Returns dict."""
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def save_config(cfg: dict) -> None:
    """Persist updated config.json to disk (pretty-printed)."""
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2, default=str)

# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

def send_telegram(message: str) -> None:
    """Send a Markdown-formatted message via Telegram Bot API."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram credentials not configured – skipping send.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        log.error(f"Telegram send failed: {e}")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create SQLite tables if they don't exist yet."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # Stores every completed trade with entry features + outcome
    cur.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol          TEXT    NOT NULL,
            entry_price     REAL    NOT NULL,
            exit_price      REAL,
            size_usd        REAL    NOT NULL,
            entry_time      TEXT    NOT NULL,
            exit_time       TEXT,
            pnl_usd         REAL,
            pnl_pct         REAL,
            status          TEXT    DEFAULT 'open',  -- open / closed
            rsi_at_entry    REAL,
            volume_chg_pct  REAL,
            btc_trend_pct   REAL,
            price_chg_4h    REAL,
            highest_price   REAL,
            trailing_active INTEGER DEFAULT 0,
            stop_loss_price REAL,
            trailing_stop_price REAL
        )
    """)
    con.commit()
    con.close()


def db_insert_trade(trade: dict) -> int:
    """Insert a new open trade. Returns the new row id."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        INSERT INTO trades
            (symbol, entry_price, size_usd, entry_time, status,
             rsi_at_entry, volume_chg_pct, btc_trend_pct, price_chg_4h,
             highest_price, stop_loss_price)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (
        trade["symbol"],
        trade["entry_price"],
        trade["size_usd"],
        trade["entry_time"],
        "open",
        trade.get("rsi_at_entry"),
        trade.get("volume_chg_pct"),
        trade.get("btc_trend_pct"),
        trade.get("price_chg_4h"),
        trade["entry_price"],          # highest_price starts at entry
        trade["stop_loss_price"],
    ))
    trade_id = cur.lastrowid
    con.commit()
    con.close()
    return trade_id


def db_update_trade_close(trade_id: int, exit_price: float,
                          exit_time: str, pnl_usd: float, pnl_pct: float) -> None:
    """Mark a trade as closed and record the PnL."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        UPDATE trades
        SET exit_price=?, exit_time=?, pnl_usd=?, pnl_pct=?, status='closed'
        WHERE id=?
    """, (exit_price, exit_time, pnl_usd, pnl_pct, trade_id))
    con.commit()
    con.close()


def db_update_trailing(trade_id: int, highest_price: float,
                       trailing_active: int, trailing_stop_price: float) -> None:
    """Update the trailing stop state for an open trade."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        UPDATE trades
        SET highest_price=?, trailing_active=?, trailing_stop_price=?
        WHERE id=?
    """, (highest_price, trailing_active, trailing_stop_price, trade_id))
    con.commit()
    con.close()


def db_get_open_trades() -> list[dict]:
    """Return all currently open trades as a list of dicts."""
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT * FROM trades WHERE status='open'")
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows


def db_get_closed_trades(min_count: int = 0) -> list[dict]:
    """Return all closed trades. Returns [] if fewer than min_count exist."""
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT * FROM trades WHERE status='closed' ORDER BY exit_time")
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    if len(rows) < min_count:
        return []
    return rows


def db_get_today_pnl() -> float:
    """Sum PnL of all trades closed today (local Berlin time)."""
    berlin = pytz.timezone("Europe/Berlin")
    today_str = datetime.now(berlin).strftime("%Y-%m-%d")
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        SELECT COALESCE(SUM(pnl_usd), 0)
        FROM trades
        WHERE status='closed' AND exit_time LIKE ?
    """, (f"{today_str}%",))
    result = cur.fetchone()[0]
    con.close()
    return float(result)

# ---------------------------------------------------------------------------
# Hyperliquid API helpers
# ---------------------------------------------------------------------------

HL_API_URL = "https://api.hyperliquid.xyz/info"
_CANDLE_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}  # symbol -> (timestamp, df)
CACHE_TTL = 60 * 14  # cache candles for ~14 minutes (refresh each scan cycle)


def hl_post(payload: dict, retries: int = 3, backoff: float = 5.0) -> dict | list | None:
    """POST to Hyperliquid info endpoint with retry logic."""
    for attempt in range(retries):
        try:
            resp = requests.post(HL_API_URL, json=payload, timeout=20)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429:
                log.warning(f"Rate limited by Hyperliquid – waiting {backoff}s (attempt {attempt+1})")
                time.sleep(backoff)
                backoff *= 2
            else:
                log.error(f"HTTP error from Hyperliquid: {e}")
                return None
        except Exception as e:
            log.error(f"API call failed (attempt {attempt+1}): {e}")
            time.sleep(backoff)
    return None


def get_all_perp_meta() -> list[dict]:
    """Fetch metadata for all perpetual markets."""
    data = hl_post({"type": "meta"})
    if data and "universe" in data:
        return data["universe"]
    return []


def get_candles(symbol: str, interval: str = "15m", lookback_hours: int = 4) -> pd.DataFrame | None:
    """
    Fetch OHLCV candles for `symbol` from Hyperliquid.
    Returns a DataFrame with columns: time, open, high, low, close, volume.
    Uses a short in-memory cache to avoid hammering the API during scans.
    """
    global _CANDLE_CACHE
    now_ts = time.time()

    # Return cached data if still fresh
    if symbol in _CANDLE_CACHE:
        cached_at, cached_df = _CANDLE_CACHE[symbol]
        if now_ts - cached_at < CACHE_TTL:
            return cached_df

    end_ms   = int(now_ts * 1000)
    start_ms = end_ms - lookback_hours * 3600 * 1000

    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
        },
    }
    raw = hl_post(payload)
    if not raw:
        return None

    try:
        df = pd.DataFrame(raw, columns=["time", "open", "high", "low", "close", "volume", "ntrades"])
        # Cast float columns first; coerce invalid values to NaN
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        # Convert time separately: drop rows where time is non-finite before int cast
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        df.dropna(subset=["time", "close", "volume"], inplace=True)
        df["time"] = df["time"].astype("int64")
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        if df.empty:
            return None
        _CANDLE_CACHE[symbol] = (now_ts, df)
        return df
    except Exception as e:
        log.error(f"Failed to parse candles for {symbol}: {e}")
        return None


def get_current_price(symbol: str) -> float | None:
    """Get the latest mark/mid price for a symbol."""
    data = hl_post({"type": "allMids"})
    if data and isinstance(data, dict) and symbol in data:
        try:
            return float(data[symbol])
        except (ValueError, TypeError):
            pass
    return None


def get_all_mids() -> dict[str, float]:
    """Return {symbol: mid_price} for all available perps."""
    data = hl_post({"type": "allMids"})
    if data and isinstance(data, dict):
        result = {}
        for k, v in data.items():
            try:
                result[k] = float(v)
            except (ValueError, TypeError):
                pass
        return result
    return {}

# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------

def calc_rsi(closes: pd.Series, period: int = 14) -> float | None:
    """Wilder RSI. Returns the last RSI value or None if not enough data."""
    if len(closes) < period + 1:
        return None
    delta = closes.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else None


def calc_volume_change_pct(volumes: pd.Series) -> float:
    """Compare last candle volume vs. mean of prior candles (%)."""
    if len(volumes) < 2:
        return 0.0
    mean_vol = volumes.iloc[:-1].mean()
    if mean_vol == 0:
        return 0.0
    return float((volumes.iloc[-1] - mean_vol) / mean_vol * 100)


def calc_price_change_pct(closes: pd.Series) -> float:
    """Percent price change from first to last close."""
    if len(closes) < 2 or closes.iloc[0] == 0:
        return 0.0
    return float((closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0] * 100)

# ---------------------------------------------------------------------------
# Scanner – find best momentum candidate
# ---------------------------------------------------------------------------

def scan_market(cfg: dict) -> dict | None:
    """
    Scan all Hyperliquid perpetuals, apply RSI filter, and return
    the best momentum candidate (or None if nothing qualifies).

    Returns a dict with: symbol, price, rsi, volume_chg_pct,
    price_chg_4h, btc_trend_pct
    """
    log.info("Starting market scan …")
    rsi_limit     = cfg["rsi_overbought_limit"]
    lookback_h    = cfg["lookback_hours"]
    rsi_period    = cfg["rsi_period"]

    # --- Fetch BTC trend as reference ---
    btc_df = get_candles("BTC", lookback_hours=lookback_h)
    btc_trend = calc_price_change_pct(btc_df["close"]) if btc_df is not None and not btc_df.empty else 0.0

    # --- Fetch all current mid prices ---
    all_mids = get_all_mids()
    if not all_mids:
        log.warning("Could not fetch mid prices – aborting scan.")
        return None

    open_trades = db_get_open_trades()
    open_symbols = {t["symbol"] for t in open_trades}

    # Score all eligible symbols
    candidates = []
    meta = get_all_perp_meta()
    symbols = [m["name"] for m in meta if m.get("name") and m["name"] != "BTC"]

    # Limit to symbols that have a mid price (i.e. actively traded)
    symbols = [s for s in symbols if s in all_mids]

    for symbol in symbols:
        if symbol in open_symbols:
            continue  # already have a position

        # Small sleep to be polite with API rate limits
        time.sleep(0.15)

        df = get_candles(symbol, lookback_hours=lookback_h)
        if df is None or len(df) < rsi_period + 2:
            continue

        rsi = calc_rsi(df["close"], period=rsi_period)
        if rsi is None or rsi > rsi_limit:
            continue  # overbought or no data

        vol_chg  = calc_volume_change_pct(df["volume"])
        price_chg = calc_price_change_pct(df["close"])

        # Composite momentum score: weight volume more, price confirms
        score = vol_chg * 0.6 + price_chg * 0.4

        if price_chg > 0 and vol_chg > 0:  # must have positive momentum on both
            candidates.append({
                "symbol":        symbol,
                "price":         all_mids[symbol],
                "rsi":           rsi,
                "volume_chg_pct": vol_chg,
                "price_chg_4h":  price_chg,
                "btc_trend_pct": btc_trend,
                "score":         score,
            })

    if not candidates:
        log.info("No qualifying momentum candidates found.")
        return None

    best = max(candidates, key=lambda x: x["score"])
    log.info(
        f"Best candidate: {best['symbol']} | RSI={best['rsi']:.1f} "
        f"| VolChg={best['volume_chg_pct']:.1f}% | PriceChg={best['price_chg_4h']:.1f}%"
    )
    return best

# ---------------------------------------------------------------------------
# Trade execution (paper only)
# ---------------------------------------------------------------------------

def open_trade(candidate: dict, cfg: dict) -> None:
    """
    Simulate opening a LONG position.
    Deducts position size from virtual balance and persists to DB.
    """
    balance    = cfg["virtual_balance_usd"]
    size_usd   = balance * cfg["position_size_pct"]
    stop_loss_pct = cfg["initial_stop_loss_pct"] / 100
    stop_price = candidate["price"] * (1 - stop_loss_pct)

    berlin    = pytz.timezone(cfg["timezone"])
    entry_ts  = datetime.now(berlin).strftime("%Y-%m-%d %H:%M:%S")

    trade = {
        "symbol":         candidate["symbol"],
        "entry_price":    candidate["price"],
        "size_usd":       size_usd,
        "entry_time":     entry_ts,
        "rsi_at_entry":   candidate["rsi"],
        "volume_chg_pct": candidate["volume_chg_pct"],
        "btc_trend_pct":  candidate["btc_trend_pct"],
        "price_chg_4h":   candidate["price_chg_4h"],
        "stop_loss_price": stop_price,
    }

    trade_id = db_insert_trade(trade)

    # Deduct position size from virtual balance
    cfg["virtual_balance_usd"] = round(balance - size_usd, 2)
    save_config(cfg)

    log.info(
        f"[TRADE OPEN] #{trade_id} {candidate['symbol']} @ ${candidate['price']:.4f} "
        f"| Size: ${size_usd:.2f} | SL: ${stop_price:.4f}"
    )

    msg = (
        f"🟢 *PAPER-TRADE:* LONG {candidate['symbol']} bei ${candidate['price']:.4f}\n"
        f"📊 Grund: RSI={candidate['rsi']:.1f} | Volumen +{candidate['volume_chg_pct']:.1f}% "
        f"| Kurs +{candidate['price_chg_4h']:.1f}% (4h)\n"
        f"💵 Positionsgröße: ${size_usd:,.2f} | Stop-Loss: ${stop_price:.4f}"
    )
    send_telegram(msg)


def close_trade(trade: dict, current_price: float, cfg: dict, reason: str = "") -> None:
    """
    Simulate closing a LONG position.
    Calculates PnL and credits it back to the virtual balance.
    """
    entry_price = trade["entry_price"]
    size_usd    = trade["size_usd"]
    pnl_pct     = (current_price - entry_price) / entry_price * 100
    pnl_usd     = size_usd * (pnl_pct / 100)
    exit_value  = size_usd + pnl_usd  # return original size + profit/loss

    berlin   = pytz.timezone(cfg["timezone"])
    exit_ts  = datetime.now(berlin).strftime("%Y-%m-%d %H:%M:%S")

    db_update_trade_close(trade["id"], current_price, exit_ts, round(pnl_usd, 2), round(pnl_pct, 4))

    # Credit exit value back to virtual balance
    cfg["virtual_balance_usd"] = round(cfg["virtual_balance_usd"] + exit_value, 2)
    cfg["total_trades_closed"] = cfg.get("total_trades_closed", 0) + 1
    if pnl_usd > 0:
        cfg["total_wins"] = cfg.get("total_wins", 0) + 1
    save_config(cfg)

    pnl_emoji = "💰" if pnl_usd >= 0 else "🔴"
    sign      = "+" if pnl_usd >= 0 else ""
    log.info(
        f"[TRADE CLOSE] #{trade['id']} {trade['symbol']} @ ${current_price:.4f} "
        f"| PnL: {sign}${pnl_usd:.2f} ({sign}{pnl_pct:.2f}%) | Reason: {reason}"
    )

    msg = (
        f"{pnl_emoji} *TRADE GESCHLOSSEN:* {trade['symbol']} LONG beendet\n"
        f"📈 Einstieg: ${entry_price:.4f} → Ausstieg: ${current_price:.4f}\n"
        f"💵 Virtueller PnL: {sign}${pnl_usd:,.2f} USD ({sign}{pnl_pct:.2f}%)\n"
        f"🏦 Neuer Kontostand: ${cfg['virtual_balance_usd']:,.2f}\n"
        f"ℹ️ Grund: {reason}"
    )
    send_telegram(msg)

# ---------------------------------------------------------------------------
# Position monitor – check stops for all open trades
# ---------------------------------------------------------------------------

def monitor_positions(cfg: dict) -> None:
    """
    Check every open trade against current market prices.
    Enforces:
      1. Hard stop-loss (initial)
      2. Trailing stop (activated once profit >= trailing_stop_activation_pct)
    """
    open_trades = db_get_open_trades()
    if not open_trades:
        return

    trail_activation = cfg["trailing_stop_activation_pct"] / 100
    trail_distance   = cfg["trailing_stop_distance_pct"] / 100

    all_mids = get_all_mids()

    for trade in open_trades:
        symbol = trade["symbol"]
        current_price = all_mids.get(symbol)
        if current_price is None:
            # Fallback: individual API call
            current_price = get_current_price(symbol)
        if current_price is None:
            log.warning(f"Could not fetch price for {symbol} – skipping check.")
            continue

        entry_price      = trade["entry_price"]
        highest_price    = trade["highest_price"] or entry_price
        trailing_active  = bool(trade["trailing_active"])
        stop_loss_price  = trade["stop_loss_price"]
        trailing_stop_p  = trade["trailing_stop_price"] or 0.0

        # Update highest price seen
        new_highest = max(highest_price, current_price)

        # --- Check trailing stop activation ---
        profit_pct = (current_price - entry_price) / entry_price
        if not trailing_active and profit_pct >= trail_activation:
            trailing_active  = True
            trailing_stop_p  = new_highest * (1 - trail_distance)
            log.info(f"Trailing stop activated for {symbol}: ${trailing_stop_p:.4f}")

        # --- Update trailing stop level if price moved higher ---
        if trailing_active:
            new_trail_stop = new_highest * (1 - trail_distance)
            if new_trail_stop > trailing_stop_p:
                trailing_stop_p = new_trail_stop

        # Persist updated trailing state
        db_update_trailing(trade["id"], new_highest, int(trailing_active), trailing_stop_p)

        # --- Evaluate exit conditions ---
        if trailing_active and current_price <= trailing_stop_p:
            close_trade(trade, current_price, cfg, reason="Trailing Stop ausgelöst")
        elif current_price <= stop_loss_price:
            close_trade(trade, current_price, cfg, reason="Stop-Loss ausgelöst")

# ---------------------------------------------------------------------------
# Main scan + trade cycle
# ---------------------------------------------------------------------------

def run_scan_cycle() -> None:
    """
    Called every 15 minutes by the scheduler.
    1. Monitor existing positions.
    2. If slot available, scan for new entry.
    """
    cfg = load_config()

    log.info("--- Scan cycle started ---")
    monitor_positions(cfg)

    # Reload config after monitor (balance may have changed after closes)
    cfg = load_config()

    open_count = len(db_get_open_trades())
    if open_count >= cfg["max_open_trades"]:
        log.info(f"Max open trades reached ({open_count}/{cfg['max_open_trades']}) – skipping scan.")
        return

    candidate = scan_market(cfg)
    if candidate:
        open_trade(candidate, cfg)

# ---------------------------------------------------------------------------
# Daily report
# ---------------------------------------------------------------------------

def send_daily_report() -> None:
    """Send a summary Telegram message at the configured daily time."""
    cfg  = load_config()
    bal  = cfg["virtual_balance_usd"]
    open_trades = db_get_open_trades()
    today_pnl   = db_get_today_pnl()
    total_closed = cfg.get("total_trades_closed", 0)
    total_wins   = cfg.get("total_wins", 0)
    win_rate     = (total_wins / total_closed * 100) if total_closed > 0 else 0.0

    open_pos_lines = ""
    if open_trades:
        all_mids = get_all_mids()
        for t in open_trades:
            cp   = all_mids.get(t["symbol"], t["entry_price"])
            upnl = (cp - t["entry_price"]) / t["entry_price"] * t["size_usd"]
            sign = "+" if upnl >= 0 else ""
            open_pos_lines += (
                f"\n  • {t['symbol']}: {sign}${upnl:.2f} "
                f"({sign}{(cp - t['entry_price']) / t['entry_price'] * 100:.2f}%)"
            )
    else:
        open_pos_lines = "\n  _(keine offenen Positionen)_"

    berlin = pytz.timezone("Europe/Berlin")
    today_sign = "+" if today_pnl >= 0 else ""
    msg = (
        f"📋 *Daily Report* – {datetime.now(berlin).strftime('%d.%m.%Y')}\n\n"
        f"🏦 *Kontostand:* ${bal:,.2f}\n"
        f"📅 *Heutiger PnL:* {today_sign}${today_pnl:,.2f}\n"
        f"📂 *Offene Positionen:* {len(open_trades)}{open_pos_lines}\n\n"
        f"📊 *Gesamt Stats:*\n"
        f"  Trades gesamt: {total_closed} | Wins: {total_wins} | Win-Rate: {win_rate:.1f}%"
    )
    send_telegram(msg)
    log.info("Daily report sent.")

# ---------------------------------------------------------------------------
# ML optimisation
# ---------------------------------------------------------------------------

def run_ml_optimization() -> None:
    """
    Weekly self-optimisation using a RandomForestRegressor.

    Features used: rsi_at_entry, volume_chg_pct, btc_trend_pct, price_chg_4h
    Target:        pnl_pct (percent profit/loss of closed trade)

    Strategy:
      - Train on all closed trades.
      - Analyse feature importances and mean outcomes at different RSI levels.
      - Nudge config parameters by up to ±1 step in the direction that improves PnL.
    """
    cfg = load_config()
    min_trades = cfg.get("ml_min_trades_required", 10)

    trades = db_get_closed_trades(min_count=min_trades)
    if not trades:
        log.info(f"ML: Not enough closed trades (need {min_trades}) – skipping optimisation.")
        send_telegram(
            f"🧠 *KI-Update (übersprungen):* Noch nicht genug Daten "
            f"(benötige mindestens {min_trades} abgeschlossene Trades)."
        )
        return

    df = pd.DataFrame(trades)

    # Drop rows with missing features
    feature_cols = ["rsi_at_entry", "volume_chg_pct", "btc_trend_pct", "price_chg_4h"]
    df = df.dropna(subset=feature_cols + ["pnl_pct"])
    if len(df) < min_trades:
        log.info("ML: Too many NaN rows – skipping.")
        return

    X = df[feature_cols].values
    y = df["pnl_pct"].values

    # Train RandomForest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importances = dict(zip(feature_cols, rf.feature_importances_))
    log.info(f"ML: Feature importances: {importances}")

    step = cfg.get("ml_adjustment_step", 1.0)

    # --- RSI limit adjustment ---
    # Split trades: high RSI vs. low RSI entries, compare mean PnL
    median_rsi = df["rsi_at_entry"].median()
    high_rsi_pnl = df[df["rsi_at_entry"] > median_rsi]["pnl_pct"].mean()
    low_rsi_pnl  = df[df["rsi_at_entry"] <= median_rsi]["pnl_pct"].mean()

    old_rsi_limit = cfg["rsi_overbought_limit"]
    new_rsi_limit = old_rsi_limit

    if high_rsi_pnl < low_rsi_pnl:
        # High RSI entries perform worse → tighten the RSI limit
        new_rsi_limit = max(cfg["rsi_limit_min"], old_rsi_limit - step)
    else:
        # High RSI entries perform OK → loosen the RSI limit slightly
        new_rsi_limit = min(cfg["rsi_limit_max"], old_rsi_limit + step)

    # --- Stop-loss adjustment ---
    # Trades that hit stop-loss vs. trailing stop
    stopped_out = df[df["pnl_pct"] <= -cfg["initial_stop_loss_pct"] + 0.5]
    mean_sl_pnl = stopped_out["pnl_pct"].mean() if not stopped_out.empty else 0.0

    old_sl = cfg["initial_stop_loss_pct"]
    new_sl = old_sl

    if mean_sl_pnl < -old_sl * 0.8 and len(stopped_out) > 2:
        # Stop-loss is being hit hard – tighten it slightly
        new_sl = max(cfg["stop_loss_min"], old_sl - step)
    elif len(stopped_out) == 0 and df["pnl_pct"].mean() > 0:
        # No stops hit and avg PnL positive – can afford to widen slightly
        new_sl = min(cfg["stop_loss_max"], old_sl + step * 0.5)

    # --- Trailing stop adjustment ---
    winners = df[df["pnl_pct"] > 0]
    old_trail = cfg["trailing_stop_distance_pct"]
    new_trail = old_trail

    if not winners.empty:
        avg_winner_pnl = winners["pnl_pct"].mean()
        if avg_winner_pnl < cfg["trailing_stop_activation_pct"]:
            # Winners are barely above activation – tighten trail to protect gains
            new_trail = max(cfg["trailing_stop_min"], old_trail - step * 0.5)
        elif avg_winner_pnl > cfg["trailing_stop_activation_pct"] * 2:
            # Big winners – can afford to let them run more
            new_trail = min(cfg["trailing_stop_max"], old_trail + step * 0.5)

    # Apply new values
    cfg["rsi_overbought_limit"]    = round(new_rsi_limit, 1)
    cfg["initial_stop_loss_pct"]   = round(new_sl, 2)
    cfg["trailing_stop_distance_pct"] = round(new_trail, 2)
    cfg["last_ml_update"]          = datetime.utcnow().isoformat()
    save_config(cfg)

    log.info(
        f"ML: RSI limit {old_rsi_limit} → {new_rsi_limit} | "
        f"SL {old_sl}% → {new_sl}% | Trail {old_trail}% → {new_trail}%"
    )

    msg = (
        f"🧠 *KI-Update abgeschlossen!*\n\n"
        f"📊 Analysierte Trades: {len(df)}\n\n"
        f"⚙️ *Neue Parameter:*\n"
        f"  • RSI-Limit: {old_rsi_limit} → *{new_rsi_limit}*\n"
        f"  • Stop-Loss: {old_sl}% → *{new_sl}%*\n"
        f"  • Trailing Stop: {old_trail}% → *{new_trail}%*\n\n"
        f"🔬 *Feature-Wichtigkeit:*\n"
        + "\n".join(f"  • {k}: {v*100:.1f}%" for k, v in sorted(importances.items(), key=lambda x: -x[1]))
    )
    send_telegram(msg)

# ---------------------------------------------------------------------------
# Scheduler setup
# ---------------------------------------------------------------------------

def _berlin_to_utc(time_str: str, tz_name: str) -> str:
    """
    Convert a HH:MM wall-clock time in `tz_name` to the UTC equivalent.
    Uses today's date so DST is handled correctly.
    The result is what `schedule` needs, since the VPS runs on UTC.
    """
    local_tz = pytz.timezone(tz_name)
    utc_tz   = pytz.utc
    now_local = datetime.now(local_tz)
    h, m = map(int, time_str.split(":"))
    local_dt = local_tz.localize(
        datetime(now_local.year, now_local.month, now_local.day, h, m, 0)
    )
    return local_dt.astimezone(utc_tz).strftime("%H:%M")


def setup_schedule(cfg: dict) -> None:
    """
    Register all recurring jobs with the `schedule` library.
    Times in config.json are Europe/Berlin – they are converted to UTC
    before being passed to `schedule`, which always runs on system time (UTC).
    """
    tz_name = cfg["timezone"]  # "Europe/Berlin"

    # --- Scan cycle every 15 minutes (no timezone conversion needed) ---
    schedule.every(cfg["scan_interval_minutes"]).minutes.do(run_scan_cycle)

    # --- Daily report: convert Berlin time → UTC ---
    daily_time_berlin = cfg.get("daily_report_time", "18:30")
    daily_time_utc    = _berlin_to_utc(daily_time_berlin, tz_name)
    schedule.every().day.at(daily_time_utc).do(send_daily_report)

    # --- Weekly ML optimisation: convert Berlin time → UTC ---
    ml_time_berlin = cfg.get("ml_optimization_time", "20:00")
    ml_time_utc    = _berlin_to_utc(ml_time_berlin, tz_name)
    schedule.every().sunday.at(ml_time_utc).do(run_ml_optimization)

    log.info(
        f"Scheduler ready: scan every {cfg['scan_interval_minutes']}min | "
        f"daily report at {daily_time_berlin} Berlin ({daily_time_utc} UTC) | "
        f"ML every Sunday at {ml_time_berlin} Berlin ({ml_time_utc} UTC)"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=== Hyperliquid Paper-Trading Bot starting ===")

    # Initialise database
    init_db()

    # Load configuration
    cfg = load_config()

    # Validate Telegram credentials
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "your_telegram_bot_token_here":
        log.warning(
            "Telegram Bot Token not set. Edit .env and add TELEGRAM_BOT_TOKEN."
        )
    if not TELEGRAM_CHAT_ID or TELEGRAM_CHAT_ID == "your_telegram_chat_id_here":
        log.warning(
            "Telegram Chat ID not set. Edit .env and add TELEGRAM_CHAT_ID."
        )

    # Send start notification
    berlin = pytz.timezone(cfg["timezone"])
    start_msg = (
        f"🤖 *System-Status:* Hyperliquid Paper-Trading-Bot gestartet.\n"
        f"💰 Startkapital geladen: *${cfg['virtual_balance_usd']:,.2f} USD*\n"
        f"🔍 Scanne alle {cfg['scan_interval_minutes']} Minuten nach Momentum-Coins.\n"
        f"⏰ Tagesreport um {cfg['daily_report_time']} Uhr | "
        f"KI-Update jeden Sonntag um {cfg['ml_optimization_time']} Uhr"
    )
    send_telegram(start_msg)

    # Register scheduled jobs
    setup_schedule(cfg)

    # Run an initial scan immediately on startup
    log.info("Running initial scan on startup …")
    run_scan_cycle()

    # Main loop – keep running and execute scheduled jobs
    log.info("Entering main scheduler loop. Press Ctrl+C to stop.")
    while True:
        try:
            schedule.run_pending()
            time.sleep(30)
        except KeyboardInterrupt:
            log.info("Bot stopped by user (KeyboardInterrupt).")
            break
        except Exception as e:
            log.error(f"Unexpected error in main loop: {e}", exc_info=True)
            # Don't crash – wait a bit, reload config, and continue
            time.sleep(60)


if __name__ == "__main__":
    main()
