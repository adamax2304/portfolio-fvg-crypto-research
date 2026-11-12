import { useState } from "react";
import { useStore } from "../store/useStore";
import { fmtUsd, fmtPrice, fmtTime, pnlColor } from "../lib/format";

export default function TradesPage() {
  const trades = useStore((s) => s.trades);
  const [filter, setFilter] = useState("all");
  const [sortBy, setSortBy] = useState("time");
  const [sortDir, setSortDir] = useState("desc");

  const filtered = trades.filter((t) => {
    if (filter === "wins") return (t.pnl_usd || 0) > 0;
    if (filter === "losses") return (t.pnl_usd || 0) <= 0;
    return true;
  });

  const sorted = [...filtered].sort((a, b) => {
    let va, vb;
    if (sortBy === "time") {
      va = new Date(a.exit_time || a.entry_time || 0).getTime();
      vb = new Date(b.exit_time || b.entry_time || 0).getTime();
    } else if (sortBy === "pnl") {
      va = a.pnl_usd || 0;
      vb = b.pnl_usd || 0;
    } else if (sortBy === "symbol") {
      va = a.symbol || "";
      vb = b.symbol || "";
    }
    return sortDir === "desc" ? vb - va : va - vb;
  });

  const toggleSort = (col) => {
    if (sortBy === col) setSortDir((d) => (d === "desc" ? "asc" : "desc"));
    else { setSortBy(col); setSortDir("desc"); }
  };

  const wins = trades.filter((t) => (t.pnl_usd || 0) > 0).length;
  const totalPnl = trades.reduce((s, t) => s + (t.pnl_usd || 0), 0);

  return (
    <div className="space-y-5 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-bold text-gray-100">Trade History</h1>
          <p className="text-xs text-gray-500 mt-0.5">
            {trades.length} trades | Win rate: {trades.length ? ((wins / trades.length) * 100).toFixed(1) : 0}% | Total PnL:{" "}
            <span className={pnlColor(totalPnl)}>{fmtUsd(totalPnl)}</span>
          </p>
        </div>
        <div className="flex gap-1">
          {["all", "wins", "losses"].map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                filter === f ? "bg-accent/15 text-accent" : "text-gray-400 hover:text-gray-200 hover:bg-dark-600/50"
              }`}
            >
              {f.charAt(0).toUpperCase() + f.slice(1)}
            </button>
          ))}
        </div>
      </div>

      <div className="card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="bg-dark-700/30">
                {[
                  { id: "time", label: "Time", align: "left" },
                  { id: "symbol", label: "Symbol", align: "left" },
                  { id: "side", label: "Side", align: "left" },
                  { id: "entry", label: "Entry", align: "right" },
                  { id: "exit", label: "Exit", align: "right" },
                  { id: "rr", label: "RR", align: "right" },
                  { id: "pnl", label: "PnL", align: "right" },
                  { id: "reason", label: "Exit Reason", align: "left" },
                ].map(({ id, label, align }) => (
                  <th
                    key={id}
                    className={`px-4 py-3 font-medium text-gray-400 cursor-pointer hover:text-gray-200 ${
                      align === "right" ? "text-right" : "text-left"
                    }`}
                    onClick={() => ["time", "pnl", "symbol"].includes(id) && toggleSort(id)}
                  >
                    {label}
                    {sortBy === id && (
                      <span className="ml-1 text-accent">{sortDir === "desc" ? "\u2193" : "\u2191"}</span>
                    )}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sorted.map((t, i) => (
                <tr key={i} className="border-b border-dark-800/50 hover:bg-dark-700/20 transition-colors">
                  <td className="px-4 py-3 text-gray-400">{fmtTime(t.exit_time || t.entry_time)}</td>
                  <td className="px-4 py-3 font-medium text-gray-200">
                    {(t.symbol || "").replace("/USDT:USDT", "")}
                  </td>
                  <td className={`px-4 py-3 font-semibold ${(t.direction || t.side) === "LONG" ? "text-profit" : "text-loss"}`}>
                    {t.direction || t.side}
                  </td>
                  <td className="px-4 py-3 text-right font-mono text-gray-300">{fmtPrice(t.entry_price)}</td>
                  <td className="px-4 py-3 text-right font-mono text-gray-300">{fmtPrice(t.exit_price)}</td>
                  <td className="px-4 py-3 text-right text-gray-300">
                    {t.risk_reward ? Number(t.risk_reward).toFixed(1) + "R" : "-"}
                  </td>
                  <td className={`px-4 py-3 text-right font-semibold ${pnlColor(t.pnl_usd)}`}>
                    {fmtUsd(t.pnl_usd)}
                  </td>
                  <td className="px-4 py-3 text-gray-400">{t.exit_reason || "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {sorted.length === 0 && (
          <div className="text-center py-8 text-gray-500 text-sm">
            {filter !== "all" ? "No matching trades" : "No trades recorded yet"}
          </div>
        )}
      </div>
    </div>
  );
}
