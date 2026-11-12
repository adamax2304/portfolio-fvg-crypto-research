"""
Paper-trading helper for the broad xsec momentum v3 configuration.

What it does:
- replays the chosen historical basket logic end-to-end
- saves per-rebalance basket snapshots
- emits the latest target basket that would be held now

This stays inside the current local research stack and uses the same cached
hourly and funding data as the v3 backtest.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


CACHE_DIR = Path(__file__).parent / "data_cache"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

START_BALANCE = 1_000.0
FEE_PER_SIDE = 0.00055
GROSS_LONG = 0.50
GROSS_SHORT = 0.50

DEFAULT_CFG = {
    "lookback_h": 24,
    "hold_h": 24,
    "top_n": 2,
    "min_abs_ret": 0.02,
    "liquid_n": 15,
    "funding_tilt": 0.0,
    "vol_lookback_h": 24,
    "corr_lookback_h": 168,
    "corr_cap": 0.55,
    "retain_extra": 1,
}


def discover_universe() -> list[str]:
    hourly = {p.stem.replace("_1h_730d", "") for p in CACHE_DIR.glob("*_1h_730d.parquet")}
    funding = {p.stem.replace("_funding_730d", "") for p in CACHE_DIR.glob("*_funding_730d.parquet")}
    return sorted(hourly & funding)


def load_hourly(base: str) -> pd.DataFrame:
    return pd.read_parquet(CACHE_DIR / f"{base}_1h_730d.parquet")[["timestamp", "close", "volume"]].astype(
        {"timestamp": "int64", "close": "float64", "volume": "float64"}
    )


def load_funding(base: str) -> pd.DataFrame:
    return pd.read_parquet(CACHE_DIR / f"{base}_funding_730d.parquet")[["timestamp", "funding_rate"]].astype(
        {"timestamp": "int64", "funding_rate": "float64"}
    )


def monthly_stats(equity_points: list[tuple[int, float]]) -> list[dict]:
    idx = pd.to_datetime([ts for ts, _ in equity_points], unit="ms", utc=True)
    ser = pd.Series([bal for _, bal in equity_points], index=idx, dtype=float).groupby(level=0).last().sort_index()
    daily_idx = pd.date_range(ser.index[0], ser.index[-1], freq="D", tz="UTC")
    daily = ser.reindex(ser.index.union(daily_idx)).sort_index().ffill().reindex(daily_idx, method="ffill").fillna(
        START_BALANCE
    )
    monthly_end = daily.resample("ME").last()
    prev = monthly_end.shift(1).fillna(START_BALANCE)
    out = []
    for ts, end_bal in monthly_end.items():
        start_bal = float(prev.loc[ts])
        ret_pct = (float(end_bal) / start_bal - 1.0) * 100.0 if start_bal > 0 else 0.0
        out.append(
            {
                "month": ts.strftime("%Y-%m"),
                "start": round(start_bal, 2),
                "end": round(float(end_bal), 2),
                "ret_pct": round(ret_pct, 2),
            }
        )
    return out


def inv_vol_weights(items: list[tuple[str, float]]) -> dict[str, float]:
    inv = [(sym, 1.0 / max(vol, 1e-6)) for sym, vol in items]
    total = sum(v for _, v in inv)
    if total <= 0:
        n = len(inv)
        return {sym: 1.0 / n for sym, _ in inv}
    return {sym: v / total for sym, v in inv}


def corr_ok(candidate: str, selected: list[str], corr_map: dict[tuple[str, str], float], corr_cap: float) -> bool:
    for sym in selected:
        pair = (candidate, sym) if candidate < sym else (sym, candidate)
        if abs(corr_map.get(pair, 0.0)) > corr_cap:
            return False
    return True


def build_side(
    candidates: list[dict],
    prev_symbols: set[str],
    top_n: int,
    retain_extra: int,
    corr_map: dict[tuple[str, str], float],
    corr_cap: float,
) -> list[dict]:
    selected: list[dict] = []
    selected_syms: list[str] = []
    retain_pool = candidates[: top_n + retain_extra]

    retained = [c for c in retain_pool if c["sym"] in prev_symbols]
    retained.sort(key=lambda x: x["score"], reverse=True)
    for cand in retained:
        if len(selected) >= top_n:
            break
        if corr_ok(cand["sym"], selected_syms, corr_map, corr_cap):
            selected.append(cand)
            selected_syms.append(cand["sym"])

    for cand in candidates:
        if len(selected) >= top_n:
            break
        if cand["sym"] in selected_syms:
            continue
        if corr_ok(cand["sym"], selected_syms, corr_map, corr_cap):
            selected.append(cand)
            selected_syms.append(cand["sym"])

    return selected


def compute_rebalance_snapshot(
    i: int,
    ref: np.ndarray,
    hourly: dict[str, pd.DataFrame],
    funding: dict[str, pd.DataFrame],
    prev_holdings: dict[str, dict],
    cfg: dict,
    allow_incomplete: bool = False,
) -> dict | None:
    t = int(ref[i])
    t_back = int(ref[i - cfg["lookback_h"]])

    sym_ts = {s: df["timestamp"].to_numpy(dtype=np.int64) for s, df in hourly.items()}
    sym_close = {s: df["close"].to_numpy(dtype=float) for s, df in hourly.items()}
    sym_vol = {s: df["volume"].to_numpy(dtype=float) for s, df in hourly.items()}

    rows = []
    corr_returns: dict[str, np.ndarray] = {}
    for sym in hourly:
        ts = sym_ts[sym]
        j = int(np.searchsorted(ts, t, side="left"))
        j_back = int(np.searchsorted(ts, t_back, side="left"))
        if j >= len(ts) or j_back >= len(ts) or ts[j] != t or ts[j_back] != t_back:
            continue
        close_arr = sym_close[sym]
        if close_arr[j_back] <= 0 or close_arr[j] <= 0:
            continue
        ret = close_arr[j] / close_arr[j_back] - 1.0
        dollar_vol = close_arr[j] * sym_vol[sym][j]

        vol_window = close_arr[j - cfg["vol_lookback_h"] : j + 1]
        if len(vol_window) < cfg["vol_lookback_h"] + 1:
            continue
        vol_lr = np.diff(np.log(vol_window))
        vol = float(np.std(vol_lr)) if len(vol_lr) else 0.0

        corr_window = close_arr[j - cfg["corr_lookback_h"] : j + 1]
        if len(corr_window) < cfg["corr_lookback_h"] + 1:
            continue
        corr_lr = np.diff(np.log(corr_window))
        corr_returns[sym] = corr_lr

        ff = funding[sym]
        mask = (ff["timestamp"] > t_back) & (ff["timestamp"] <= t)
        fsum = float(ff.loc[mask, "funding_rate"].sum()) if mask.any() else 0.0
        long_score = ret - cfg["funding_tilt"] * max(fsum, 0.0)
        short_score = -ret + cfg["funding_tilt"] * max(fsum, 0.0)
        rows.append(
            {
                "sym": sym,
                "ret": ret,
                "dollar_vol": dollar_vol,
                "vol": vol,
                "fsum": fsum,
                "long_score": long_score,
                "short_score": short_score,
            }
        )

    if len(rows) < cfg["top_n"] * 2 + 5:
        if allow_incomplete:
            return {
                "timestamp_ms": t,
                "timestamp_utc": datetime.fromtimestamp(t / 1000.0, tz=timezone.utc).isoformat(),
                "basket": [],
                "target_weights": {},
                "turnover": 0.0,
                "liquid_universe": [],
                "diagnostics": {
                    "reason": "insufficient_rows",
                    "rows": len(rows),
                },
            }
        return None

    rows.sort(key=lambda x: x["dollar_vol"], reverse=True)
    liquid = rows[: cfg["liquid_n"]]
    long_candidates = [dict(r, score=r["long_score"]) for r in liquid if r["ret"] >= cfg["min_abs_ret"]]
    short_candidates = [dict(r, score=r["short_score"]) for r in liquid if r["ret"] <= -cfg["min_abs_ret"]]
    long_candidates.sort(key=lambda x: x["score"], reverse=True)
    short_candidates.sort(key=lambda x: x["score"], reverse=True)
    if len(long_candidates) < cfg["top_n"] or len(short_candidates) < cfg["top_n"]:
        if allow_incomplete:
            return {
                "timestamp_ms": t,
                "timestamp_utc": datetime.fromtimestamp(t / 1000.0, tz=timezone.utc).isoformat(),
                "basket": [],
                "target_weights": {},
                "turnover": 0.0,
                "liquid_universe": [r["sym"] for r in liquid],
                "diagnostics": {
                    "reason": "insufficient_candidates",
                    "long_candidates": len(long_candidates),
                    "short_candidates": len(short_candidates),
                    "top_longs": [
                        {
                            "symbol": x["sym"],
                            "ret_lookback_pct": round(x["ret"] * 100.0, 4),
                            "score": round(x["score"], 6),
                        }
                        for x in long_candidates[:5]
                    ],
                    "top_shorts": [
                        {
                            "symbol": x["sym"],
                            "ret_lookback_pct": round(x["ret"] * 100.0, 4),
                            "score": round(x["score"], 6),
                        }
                        for x in short_candidates[:5]
                    ],
                },
            }
        return None

    corr_map: dict[tuple[str, str], float] = {}
    liquid_syms = [r["sym"] for r in liquid if r["sym"] in corr_returns]
    for a_idx in range(len(liquid_syms)):
        for b_idx in range(a_idx + 1, len(liquid_syms)):
            a = liquid_syms[a_idx]
            b = liquid_syms[b_idx]
            ra = corr_returns[a]
            rb = corr_returns[b]
            corr = float(np.corrcoef(ra, rb)[0, 1]) if len(ra) == len(rb) and len(ra) > 3 else 0.0
            if np.isnan(corr):
                corr = 0.0
            corr_map[(a, b)] = corr

    prev_longs = {sym for sym, pos in prev_holdings.items() if pos["side"] == "LONG"}
    prev_shorts = {sym for sym, pos in prev_holdings.items() if pos["side"] == "SHORT"}
    selected_longs = build_side(
        long_candidates, prev_longs, cfg["top_n"], cfg["retain_extra"], corr_map, cfg["corr_cap"]
    )
    selected_shorts = build_side(
        short_candidates, prev_shorts, cfg["top_n"], cfg["retain_extra"], corr_map, cfg["corr_cap"]
    )
    if len(selected_longs) < cfg["top_n"] or len(selected_shorts) < cfg["top_n"]:
        if allow_incomplete:
            return {
                "timestamp_ms": t,
                "timestamp_utc": datetime.fromtimestamp(t / 1000.0, tz=timezone.utc).isoformat(),
                "basket": [],
                "target_weights": {},
                "turnover": 0.0,
                "liquid_universe": [r["sym"] for r in liquid],
                "diagnostics": {
                    "reason": "selection_filtered_by_correlation",
                    "selected_longs": len(selected_longs),
                    "selected_shorts": len(selected_shorts),
                    "corr_cap": cfg["corr_cap"],
                },
            }
        return None

    long_w = inv_vol_weights([(x["sym"], x["vol"]) for x in selected_longs])
    short_w = inv_vol_weights([(x["sym"], x["vol"]) for x in selected_shorts])

    basket = []
    target_weights: dict[str, dict] = {}
    for item in selected_longs:
        weight = GROSS_LONG * long_w[item["sym"]]
        retained = item["sym"] in prev_longs
        basket.append(
            {
                "symbol": item["sym"],
                "side": "LONG",
                "weight": weight,
                "ret_lookback_pct": item["ret"] * 100.0,
                "recent_funding_pct": item["fsum"] * 100.0,
                "realized_vol_pct": item["vol"] * 100.0,
                "score": item["score"],
                "retained": retained,
            }
        )
        target_weights[item["sym"]] = {"side": "LONG", "weight": weight}
    for item in selected_shorts:
        weight = GROSS_SHORT * short_w[item["sym"]]
        retained = item["sym"] in prev_shorts
        basket.append(
            {
                "symbol": item["sym"],
                "side": "SHORT",
                "weight": weight,
                "ret_lookback_pct": item["ret"] * 100.0,
                "recent_funding_pct": item["fsum"] * 100.0,
                "realized_vol_pct": item["vol"] * 100.0,
                "score": item["score"],
                "retained": retained,
            }
        )
        target_weights[item["sym"]] = {"side": "SHORT", "weight": weight}

    basket.sort(key=lambda x: (x["side"], x["weight"]), reverse=True)

    prev_signed = {}
    for sym, pos in prev_holdings.items():
        prev_signed[sym] = pos["weight"] if pos["side"] == "LONG" else -pos["weight"]
    new_signed = {}
    for sym, pos in target_weights.items():
        new_signed[sym] = pos["weight"] if pos["side"] == "LONG" else -pos["weight"]
    all_syms = set(prev_signed) | set(new_signed)
    turnover = sum(abs(new_signed.get(sym, 0.0) - prev_signed.get(sym, 0.0)) for sym in all_syms)

    return {
        "timestamp_ms": t,
        "timestamp_utc": datetime.fromtimestamp(t / 1000.0, tz=timezone.utc).isoformat(),
        "basket": basket,
        "target_weights": target_weights,
        "turnover": turnover,
        "liquid_universe": [r["sym"] for r in liquid],
    }


def run_history(hourly: dict[str, pd.DataFrame], funding: dict[str, pd.DataFrame], cfg: dict) -> dict:
    ref = hourly["BTC_USDT_USDT"]["timestamp"].to_numpy(dtype=np.int64)
    sym_ts = {s: df["timestamp"].to_numpy(dtype=np.int64) for s, df in hourly.items()}
    sym_close = {s: df["close"].to_numpy(dtype=float) for s, df in hourly.items()}

    balance = START_BALANCE
    start_i = max(cfg["lookback_h"], cfg["vol_lookback_h"], cfg["corr_lookback_h"])
    equity = [(int(ref[start_i]), balance)]
    rebalance_pnls: list[float] = []
    turnover_series: list[float] = []
    holdings: dict[str, dict] = {}
    snapshots: list[dict] = []

    for i in range(start_i, len(ref) - cfg["hold_h"], cfg["hold_h"]):
        t = int(ref[i])
        t_exit = int(ref[i + cfg["hold_h"]])

        if holdings:
            pnl_interval = 0.0
            for sym, pos in holdings.items():
                ts = sym_ts[sym]
                j_now = int(np.searchsorted(ts, t, side="left"))
                j_exit = int(np.searchsorted(ts, t_exit, side="left"))
                if j_now >= len(ts) or j_exit >= len(ts) or ts[j_now] != t or ts[j_exit] != t_exit:
                    continue
                entry_price = sym_close[sym][j_now]
                exit_price = sym_close[sym][j_exit]
                notional = balance * abs(pos["weight"])
                qty = notional / entry_price if entry_price > 0 else 0.0
                side_sign = 1.0 if pos["side"] == "LONG" else -1.0
                gross = (exit_price - entry_price) * qty * side_sign
                ff = funding[sym]
                mask = (ff["timestamp"] > t) & (ff["timestamp"] <= t_exit)
                funding_sum = float(ff.loc[mask, "funding_rate"].sum()) if mask.any() else 0.0
                funding_pnl = -notional * funding_sum if side_sign > 0 else notional * funding_sum
                pnl_interval += gross + funding_pnl
            balance += pnl_interval
            rebalance_pnls.append(pnl_interval)
            equity.append((t_exit, balance))

        snap = compute_rebalance_snapshot(i, ref, hourly, funding, holdings, cfg)
        if snap is None:
            holdings = {}
            continue

        turnover = snap["turnover"]
        turnover_series.append(turnover)
        balance -= balance * turnover * FEE_PER_SIDE
        snap["balance_after_fees"] = round(balance, 2)
        snapshots.append(snap)
        holdings = snap["target_weights"]

    if equity[-1][0] < int(ref[-1]):
        equity.append((int(ref[-1]), balance))

    wins = sum(1 for x in rebalance_pnls if x > 0)
    gross_profit = sum(x for x in rebalance_pnls if x > 0)
    gross_loss = -sum(x for x in rebalance_pnls if x < 0)
    pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    peak = START_BALANCE
    max_dd_pct = 0.0
    for _, bal in equity:
        peak = max(peak, bal)
        if peak > 0:
            max_dd_pct = max(max_dd_pct, (peak - bal) / peak * 100.0)

    monthly = monthly_stats(equity)
    return {
        "metrics": {
            "rebalances": len(rebalance_pnls),
            "wr_pct": round((wins / len(rebalance_pnls) * 100.0) if rebalance_pnls else 0.0, 2),
            "profit_factor": round(float(pf), 3) if np.isfinite(pf) else None,
            "return_pct": round((balance / START_BALANCE - 1.0) * 100.0, 2),
            "max_dd_pct": round(max_dd_pct, 2),
            "final_balance": round(balance, 2),
            "avg_turnover": round(float(np.mean(turnover_series)) if turnover_series else 0.0, 4),
            "profitable_months": sum(1 for m in monthly if m["ret_pct"] > 0),
            "months": len(monthly),
        },
        "monthly_returns": monthly,
        "snapshots": snapshots,
        "equity_points": equity,
        "ref": ref,
        "start_i": start_i,
        "last_target_weights": holdings,
    }


def main() -> None:
    universe = discover_universe()
    hourly = {base: load_hourly(base) for base in universe}
    funding = {base: load_funding(base) for base in universe}
    cfg = dict(DEFAULT_CFG)

    hist = run_history(hourly, funding, cfg)
    ref = hist["ref"]
    latest_i = hist["start_i"] + ((len(ref) - 1 - hist["start_i"]) // cfg["hold_h"]) * cfg["hold_h"]
    latest_eval_ts = int(ref[latest_i])
    if hist["snapshots"] and hist["snapshots"][-1]["timestamp_ms"] < latest_eval_ts:
        prev_holdings = hist["last_target_weights"]
        previous_basket = hist["snapshots"][-1]["basket"]
    elif len(hist["snapshots"]) >= 2:
        prev_holdings = hist["snapshots"][-2]["target_weights"]
        previous_basket = hist["snapshots"][-2]["basket"]
    else:
        prev_holdings = {}
        previous_basket = []
    latest = compute_rebalance_snapshot(int(latest_i), ref, hourly, funding, prev_holdings, cfg, allow_incomplete=True)
    if latest is None:
        raise RuntimeError("Latest signal diagnostics unexpectedly returned None.")
    latest["status"] = "active_basket" if latest["basket"] else "flat_no_signal"

    hours_since_rebalance = int((len(ref) - 1) - latest_i)
    hours_to_next_rebalance = 0 if hours_since_rebalance == 0 else cfg["hold_h"] - hours_since_rebalance

    latest["hours_since_rebalance"] = hours_since_rebalance
    latest["hours_to_next_rebalance"] = hours_to_next_rebalance
    latest["latest_cache_bar_utc"] = datetime.fromtimestamp(int(ref[-1]) / 1000.0, tz=timezone.utc).isoformat()
    latest["previous_basket"] = previous_basket

    compact_rows = []
    for snap in hist["snapshots"]:
        for leg in snap["basket"]:
            compact_rows.append(
                {
                    "timestamp_utc": snap["timestamp_utc"],
                    "symbol": leg["symbol"],
                    "side": leg["side"],
                    "weight": round(leg["weight"], 6),
                    "ret_lookback_pct": round(leg["ret_lookback_pct"], 4),
                    "recent_funding_pct": round(leg["recent_funding_pct"], 5),
                    "realized_vol_pct": round(leg["realized_vol_pct"], 4),
                    "score": round(leg["score"], 6),
                    "retained": bool(leg["retained"]),
                    "turnover": round(snap["turnover"], 6),
                }
            )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"xsec_v3_paper_rebalances_{stamp}.csv"
    pd.DataFrame(compact_rows).to_csv(csv_path, index=False)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "assumptions": {
            "start_balance": START_BALANCE,
            "fee_per_side": FEE_PER_SIDE,
            "weighting": "inverse vol within each side",
            "gross_long": GROSS_LONG,
            "gross_short": GROSS_SHORT,
            "universe_size": len(universe),
            "universe": universe,
        },
        "historical_validation": {
            "metrics": hist["metrics"],
            "monthly_returns": hist["monthly_returns"],
            "rebalance_csv": str(csv_path),
        },
        "latest_signal": latest,
    }

    out_path = RESULTS_DIR / f"xsec_v3_paper_runner_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report_path": str(out_path), "latest_signal": latest, "metrics": hist["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
