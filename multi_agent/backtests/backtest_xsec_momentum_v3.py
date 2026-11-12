"""
Cross-sectional momentum v3:
- stateful long/short basket
- inverse-vol weights
- funding-aware scoring
- correlation caps
- turnover-aware rebalancing costs
- retention buffer to reduce needless rotation
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
    daily = ser.reindex(ser.index.union(daily_idx)).sort_index().ffill().reindex(daily_idx, method="ffill").fillna(START_BALANCE)
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
    side: str,
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


def run_cfg(
    lookback_h: int,
    hold_h: int,
    top_n: int,
    min_abs_ret: float,
    liquid_n: int,
    funding_tilt: float,
    vol_lookback_h: int,
    corr_lookback_h: int,
    corr_cap: float,
    retain_extra: int,
    hourly: dict[str, pd.DataFrame],
    funding: dict[str, pd.DataFrame],
) -> dict:
    ref = hourly["BTC_USDT_USDT"]["timestamp"].to_numpy(dtype=np.int64)
    sym_ts = {s: df["timestamp"].to_numpy(dtype=np.int64) for s, df in hourly.items()}
    sym_close = {s: df["close"].to_numpy(dtype=float) for s, df in hourly.items()}
    sym_vol = {s: df["volume"].to_numpy(dtype=float) for s, df in hourly.items()}

    balance = START_BALANCE
    start_i = max(lookback_h, vol_lookback_h, corr_lookback_h)
    equity = [(int(ref[start_i]), balance)]
    rebalance_pnls: list[float] = []
    turnover_series: list[float] = []
    holdings: dict[str, dict] = {}

    for i in range(start_i, len(ref) - hold_h, hold_h):
        t = ref[i]
        t_back = ref[i - lookback_h]
        t_exit = ref[i + hold_h]

        # Realize PnL on current holdings over the next holding interval.
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
            equity.append((int(t_exit), balance))

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

            vol_window = close_arr[j - vol_lookback_h : j + 1]
            if len(vol_window) < vol_lookback_h + 1:
                continue
            vol_lr = np.diff(np.log(vol_window))
            vol = float(np.std(vol_lr)) if len(vol_lr) else 0.0

            corr_window = close_arr[j - corr_lookback_h : j + 1]
            if len(corr_window) < corr_lookback_h + 1:
                continue
            corr_lr = np.diff(np.log(corr_window))
            corr_returns[sym] = corr_lr

            ff = funding[sym]
            mask = (ff["timestamp"] > t_back) & (ff["timestamp"] <= t)
            fsum = float(ff.loc[mask, "funding_rate"].sum()) if mask.any() else 0.0
            long_score = ret - funding_tilt * max(fsum, 0.0)
            short_score = -ret + funding_tilt * max(fsum, 0.0)
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

        if len(rows) < top_n * 2 + 5:
            holdings = {}
            continue

        rows.sort(key=lambda x: x["dollar_vol"], reverse=True)
        liquid = rows[:liquid_n]
        long_candidates = [dict(r, score=r["long_score"]) for r in liquid if r["ret"] >= min_abs_ret]
        short_candidates = [dict(r, score=r["short_score"]) for r in liquid if r["ret"] <= -min_abs_ret]
        long_candidates.sort(key=lambda x: x["score"], reverse=True)
        short_candidates.sort(key=lambda x: x["score"], reverse=True)
        if len(long_candidates) < top_n or len(short_candidates) < top_n:
            holdings = {}
            continue

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

        prev_longs = {sym for sym, pos in holdings.items() if pos["side"] == "LONG"}
        prev_shorts = {sym for sym, pos in holdings.items() if pos["side"] == "SHORT"}
        selected_longs = build_side("LONG", long_candidates, prev_longs, top_n, retain_extra, corr_map, corr_cap)
        selected_shorts = build_side("SHORT", short_candidates, prev_shorts, top_n, retain_extra, corr_map, corr_cap)
        if len(selected_longs) < top_n or len(selected_shorts) < top_n:
            holdings = {}
            continue

        long_w = inv_vol_weights([(x["sym"], x["vol"]) for x in selected_longs])
        short_w = inv_vol_weights([(x["sym"], x["vol"]) for x in selected_shorts])
        target_weights: dict[str, dict] = {}
        for item in selected_longs:
            target_weights[item["sym"]] = {"side": "LONG", "weight": GROSS_LONG * long_w[item["sym"]]}
        for item in selected_shorts:
            target_weights[item["sym"]] = {"side": "SHORT", "weight": GROSS_SHORT * short_w[item["sym"]]}

        prev_signed = {}
        for sym, pos in holdings.items():
            prev_signed[sym] = pos["weight"] if pos["side"] == "LONG" else -pos["weight"]
        new_signed = {}
        for sym, pos in target_weights.items():
            new_signed[sym] = pos["weight"] if pos["side"] == "LONG" else -pos["weight"]
        all_syms = set(prev_signed) | set(new_signed)
        turnover = sum(abs(new_signed.get(sym, 0.0) - prev_signed.get(sym, 0.0)) for sym in all_syms)
        turnover_series.append(turnover)
        balance -= balance * turnover * FEE_PER_SIDE

        holdings = target_weights

    # Extend the curve to the end of the reference sample even if the strategy
    # goes flat for long periods; otherwise monthly stats get truncated.
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
        "params": {
            "lookback_h": lookback_h,
            "hold_h": hold_h,
            "top_n": top_n,
            "min_abs_ret": min_abs_ret,
            "liquid_n": liquid_n,
            "funding_tilt": funding_tilt,
            "vol_lookback_h": vol_lookback_h,
            "corr_lookback_h": corr_lookback_h,
            "corr_cap": corr_cap,
            "retain_extra": retain_extra,
        },
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
    }


def main() -> None:
    universe = discover_universe()
    hourly = {base: load_hourly(base) for base in universe}
    funding = {base: load_funding(base) for base in universe}

    results = []
    configs = [
        (24, 8, 4, 0.02, 15, 150.0),
        (24, 24, 2, 0.02, 15, 0.0),
    ]
    for lookback_h, hold_h, top_n, min_abs_ret, liquid_n, funding_tilt in configs:
        for corr_cap in (0.55, 0.70):
            for retain_extra in (1, 3):
                results.append(
                    run_cfg(
                        lookback_h=lookback_h,
                        hold_h=hold_h,
                        top_n=top_n,
                        min_abs_ret=min_abs_ret,
                        liquid_n=liquid_n,
                        funding_tilt=funding_tilt,
                        vol_lookback_h=24,
                        corr_lookback_h=168,
                        corr_cap=corr_cap,
                        retain_extra=retain_extra,
                        hourly=hourly,
                        funding=funding,
                    )
                )

    results.sort(
        key=lambda x: (
            x["metrics"]["profit_factor"] if x["metrics"]["profit_factor"] is not None else 0.0,
            x["metrics"]["return_pct"],
        ),
        reverse=True,
    )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "assumptions": {
            "start_balance": START_BALANCE,
            "fee_per_side": FEE_PER_SIDE,
            "weighting": "inverse vol within each side",
            "rebalancing": "stateful basket with turnover-based fees instead of full round-trip each rebalance",
            "correlation_cap": "absolute hourly return correlation within same side over trailing corr_lookback_h",
            "retention": "previous holdings may stay if still within top_n + retain_extra candidates",
            "universe_size": len(universe),
            "universe": universe,
        },
        "results": results,
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"xsec_momentum_v3_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report_path": str(out_path), "top_results": results[:10]}, indent=2))


if __name__ == "__main__":
    main()
