import { useMemo, useState } from "react";
import { clsPnL, fmtPrice, fmtTime, fmtUsd } from "../lib/format";

export default function TradeHistoryTable({ rows }) {
  const [sortKey, setSortKey] = useState("exit_time");
  const [desc, setDesc] = useState(true);

  const sorted = useMemo(() => {
    const values = [...(rows || [])];
    values.sort((a, b) => {
      const av = a?.[sortKey];
      const bv = b?.[sortKey];
      if (typeof av === "number" || typeof bv === "number") {
        return (Number(av || 0) - Number(bv || 0)) * (desc ? -1 : 1);
      }
      return String(av || "").localeCompare(String(bv || "")) * (desc ? -1 : 1);
    });
    return values;
  }, [rows, sortKey, desc]);

  const toggleSort = (key) => {
    if (sortKey === key) setDesc((value) => !value);
    else {
      setSortKey(key);
      setDesc(true);
    }
  };

  return (
    <div className="glass-panel rounded-2xl p-3">
      <div className="mb-2 text-sm font-semibold text-gray-100">Trade History</div>
      <div className="scroll-y max-h-[360px] overflow-x-auto rounded-xl border border-white/10">
        <table className="min-w-full text-xs">
          <thead className="sticky top-0 bg-[#141417] text-gray-300">
            <tr>
              <th className="px-2 py-2 text-left">Symbol</th>
              <th className="px-2 py-2 text-left">Side</th>
              <th className="cursor-pointer px-2 py-2 text-left" onClick={() => toggleSort("entry_time")}>
                Entry
              </th>
              <th className="cursor-pointer px-2 py-2 text-left" onClick={() => toggleSort("exit_time")}>
                Exit
              </th>
              <th className="px-2 py-2 text-left">Entry Px</th>
              <th className="px-2 py-2 text-left">Exit Px</th>
              <th className="cursor-pointer px-2 py-2 text-left" onClick={() => toggleSort("rr")}>
                RR
              </th>
              <th className="cursor-pointer px-2 py-2 text-left" onClick={() => toggleSort("pnl_usd")}>
                PnL
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((trade) => (
              <tr key={trade.trade_id} className="border-t border-white/5 text-gray-300">
                <td className="px-2 py-2">{trade.short_symbol}</td>
                <td className="px-2 py-2">{trade.direction}</td>
                <td className="px-2 py-2">{fmtTime(trade.entry_time)}</td>
                <td className="px-2 py-2">{fmtTime(trade.exit_time)}</td>
                <td className="px-2 py-2">{fmtPrice(trade.entry_price)}</td>
                <td className="px-2 py-2">{fmtPrice(trade.exit_price)}</td>
                <td className="px-2 py-2">{Number(trade.rr || 0).toFixed(2)}</td>
                <td className={`px-2 py-2 font-semibold ${clsPnL(trade.pnl_usd)}`}>{fmtUsd(trade.pnl_usd)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

