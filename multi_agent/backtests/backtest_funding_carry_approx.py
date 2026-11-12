"""
Approximate funding carry research on local perp funding history.

This is NOT a full spot/perp arbitrage simulator because the workspace does not
currently contain historical spot data or historical basis snapshots.

What it approximates:
- open a market-neutral short-perp / long-spot style carry sleeve whenever
  perp funding is sufficiently positive
- earn/payout the actual cached perp funding stream
- pay configurable round-trip trading costs

Use this only to estimate whether the funding stream is large enough to matter.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


CACHE_DIR = Path(__file__).parent / "data_cache"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

START_BALANCE = 1_000.0


def discover_universe() -> list[str]:
    return sorted(p.stem.replace("_funding_730d", "") for p in CACHE_DIR.glob("*_funding_730d.parquet"))


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


def run_cfg(
    entry_thr: float,
    exit_thr: float,
    max_pos: int,
    roundtrip_fee: float,
    capital_mult: float,
    funding_map: dict[str, pd.DataFrame],
) -> dict:
    ref = funding_map["BTC_USDT_USDT"]["timestamp"].to_numpy(dtype="int64")
    rate_maps = {sym: df.set_index("timestamp")["funding_rate"] for sym, df in funding_map.items()}

    balance = START_BALANCE
    open_pos: dict[str, dict] = {}
    equity = [(int(ref[0]), balance)]

    for t in ref:
        funding_pnl = 0.0
        for sym, pos in list(open_pos.items()):
            rate = rate_maps[sym].get(t, None)
            if rate is not None:
                funding_pnl += pos["notional"] * float(rate)
        balance += funding_pnl

        for sym in list(open_pos.keys()):
            rate = rate_maps[sym].get(t, None)
            if rate is None or float(rate) <= exit_thr:
                balance -= open_pos[sym]["notional"] * (roundtrip_fee / 2.0)
                del open_pos[sym]

        room = max_pos - len(open_pos)
        if room > 0:
            candidates = []
            for sym in funding_map:
                if sym in open_pos:
                    continue
                rate = rate_maps[sym].get(t, None)
                if rate is None:
                    continue
                rate = float(rate)
                if rate >= entry_thr:
                    candidates.append((sym, rate))
            candidates.sort(key=lambda x: x[1], reverse=True)
            for sym, _ in candidates[:room]:
                alloc_capital = balance / max_pos
                notional = alloc_capital / capital_mult
                balance -= notional * (roundtrip_fee / 2.0)
                open_pos[sym] = {"notional": notional}

        equity.append((int(t), balance))

    for pos in open_pos.values():
        balance -= pos["notional"] * (roundtrip_fee / 2.0)
    equity.append((int(ref[-1]), balance))

    peak = START_BALANCE
    max_dd_pct = 0.0
    for _, bal in equity:
        peak = max(peak, bal)
        if peak > 0:
            max_dd_pct = max(max_dd_pct, (peak - bal) / peak * 100.0)

    monthly = monthly_stats(equity)
    return {
        "params": {
            "entry_thr": entry_thr,
            "exit_thr": exit_thr,
            "max_pos": max_pos,
            "roundtrip_fee": roundtrip_fee,
            "capital_mult": capital_mult,
        },
        "metrics": {
            "return_pct": round((balance / START_BALANCE - 1.0) * 100.0, 2),
            "max_dd_pct": round(max_dd_pct, 4),
            "final_balance": round(balance, 2),
            "profitable_months": sum(1 for m in monthly if m["ret_pct"] > 0),
            "months": len(monthly),
            "avg_monthly_pct": round(sum(m["ret_pct"] for m in monthly) / len(monthly), 3) if monthly else 0.0,
        },
        "monthly_returns": monthly,
    }


def main() -> None:
    universe = discover_universe()
    funding_map = {base: load_funding(base) for base in universe}

    results = []
    for entry_thr in (0.00008, 0.00010, 0.00012, 0.00015):
        for exit_thr in (0.0, 0.00003, 0.00005):
            for max_pos in (3, 5):
                for roundtrip_fee in (0.0010, 0.0015, 0.0020):
                    results.append(run_cfg(entry_thr, exit_thr, max_pos, roundtrip_fee, 1.2, funding_map))

    results.sort(key=lambda x: (x["metrics"]["return_pct"], -x["metrics"]["max_dd_pct"]), reverse=True)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "assumptions": {
            "start_balance": START_BALANCE,
            "approximation": "funding stream only; assumes spot hedge exists and basis risk is ignored",
            "universe_size": len(universe),
            "universe": universe,
        },
        "results": results,
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"funding_carry_approx_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report_path": str(out_path), "top_results": results[:10]}, indent=2))


if __name__ == "__main__":
    main()
