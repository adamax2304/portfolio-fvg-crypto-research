"""
Cross-sectional momentum v2:
- long/short basket
- inverse-vol weighting
- optional funding-aware score adjustment

This extends the phase-1 xsec momentum runner with more portfolio-like sizing.
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


def inv_vol_weights(items: list[tuple[str, float, float, int]]) -> dict[str, float]:
    inv = []
    for sym, _, vol, _ in items:
        inv.append((sym, 1.0 / max(vol, 1e-6)))
    total = sum(v for _, v in inv)
    if total <= 0:
        n = len(inv)
        return {sym: 1.0 / n for sym, _ in inv}
    return {sym: val / total for sym, val in inv}


def run_cfg(
    lookback_h: int,
    hold_h: int,
    top_n: int,
    min_abs_ret: float,
    liquid_n: int,
    funding_tilt: float,
    vol_lookback_h: int,
    hourly: dict[str, pd.DataFrame],
    funding: dict[str, pd.DataFrame],
) -> dict:
    ref = hourly["BTC_USDT_USDT"]["timestamp"].to_numpy(dtype=np.int64)
    sym_ts = {s: df["timestamp"].to_numpy(dtype=np.int64) for s, df in hourly.items()}
    sym_close = {s: df["close"].to_numpy(dtype=float) for s, df in hourly.items()}
    sym_vol = {s: df["volume"].to_numpy(dtype=float) for s, df in hourly.items()}
    rate_maps = {s: df.set_index("timestamp")["funding_rate"] for s, df in funding.items()}

    balance = START_BALANCE
    equity = [(int(ref[max(lookback_h, vol_lookback_h)]), balance)]
    basket_pnls: list[float] = []

    for i in range(max(lookback_h, vol_lookback_h), len(ref) - hold_h, hold_h):
        t = ref[i]
        t_back = ref[i - lookback_h]
        t_exit = ref[i + hold_h]
        rows = []
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
            window = close_arr[j - vol_lookback_h : j + 1]
            if len(window) < vol_lookback_h + 1:
                continue
            lr = np.diff(np.log(window))
            vol = float(np.std(lr)) if len(lr) else 0.0
            fsum = 0.0
            ff = funding[sym]
            mask = (ff["timestamp"] > t_back) & (ff["timestamp"] <= t)
            if mask.any():
                fsum = float(ff.loc[mask, "funding_rate"].sum())
            long_score = ret - funding_tilt * max(fsum, 0.0)
            short_score = -ret + funding_tilt * max(fsum, 0.0)
            rows.append((sym, ret, dollar_vol, j, vol, fsum, long_score, short_score))
        if len(rows) < top_n * 2 + 5:
            continue

        rows.sort(key=lambda x: x[2], reverse=True)
        liquid = rows[:liquid_n]
        longs = sorted([r for r in liquid if r[1] >= min_abs_ret], key=lambda x: x[6], reverse=True)[:top_n]
        shorts = sorted([r for r in liquid if r[1] <= -min_abs_ret], key=lambda x: x[7], reverse=True)[:top_n]
        if len(longs) < top_n or len(shorts) < top_n:
            continue

        long_weights = inv_vol_weights([(sym, ret, vol, j) for sym, ret, _, j, vol, _, _, _ in longs])
        short_weights = inv_vol_weights([(sym, ret, vol, j) for sym, ret, _, j, vol, _, _, _ in shorts])

        basket_pnl = 0.0
        for picks, side_sign, weights in ((longs, 1.0, long_weights), (shorts, -1.0, short_weights)):
            gross_capital = balance * 0.5
            for sym, _, _, j, _, _, _, _ in picks:
                alloc = gross_capital * weights[sym]
                entry = sym_close[sym][j]
                exit_price = sym_close[sym][j + hold_h]
                qty = alloc / entry
                gross = (exit_price - entry) * qty * side_sign
                fees = alloc * FEE_PER_SIDE * 2.0
                ff = funding[sym]
                mask = (ff["timestamp"] > t) & (ff["timestamp"] <= t_exit)
                funding_sum = float(ff.loc[mask, "funding_rate"].sum()) if mask.any() else 0.0
                funding_pnl = -alloc * funding_sum if side_sign > 0 else alloc * funding_sum
                basket_pnl += gross - fees + funding_pnl

        balance += basket_pnl
        basket_pnls.append(basket_pnl)
        equity.append((int(t_exit), balance))

    wins = sum(1 for x in basket_pnls if x > 0)
    gross_profit = sum(x for x in basket_pnls if x > 0)
    gross_loss = -sum(x for x in basket_pnls if x < 0)
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
        },
        "metrics": {
            "trades": len(basket_pnls),
            "wr_pct": round((wins / len(basket_pnls) * 100.0) if basket_pnls else 0.0, 2),
            "profit_factor": round(float(pf), 3) if np.isfinite(pf) else None,
            "return_pct": round((balance / START_BALANCE - 1.0) * 100.0, 2),
            "max_dd_pct": round(max_dd_pct, 2),
            "final_balance": round(balance, 2),
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
    for cfg in (
        (24, 8, 4, 0.02),
        (24, 24, 2, 0.02),
        (24, 24, 3, 0.01),
    ):
        lookback_h, hold_h, top_n, min_abs_ret = cfg
        for funding_tilt in (0.0, 50.0, 100.0, 150.0):
            results.append(
                run_cfg(
                    lookback_h=lookback_h,
                    hold_h=hold_h,
                    top_n=top_n,
                    min_abs_ret=min_abs_ret,
                    liquid_n=15,
                    funding_tilt=funding_tilt,
                    vol_lookback_h=24,
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
            "weighting": "inverse realized volatility within each side of the basket",
            "funding_tilt": "long score penalizes positive recent funding; short score rewards positive recent funding",
            "universe_size": len(universe),
            "universe": universe,
        },
        "results": results,
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"xsec_momentum_v2_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report_path": str(out_path), "top_results": results[:10]}, indent=2))


if __name__ == "__main__":
    main()
