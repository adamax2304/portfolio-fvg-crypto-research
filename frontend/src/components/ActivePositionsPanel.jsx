import { motion } from "framer-motion";
import { clsPnL, fmtPct, fmtPrice, fmtUsd } from "../lib/format";

export default function ActivePositionsPanel({ positions }) {
  if (!positions?.length) {
    return (
      <div className="rounded-2xl border border-white/10 bg-white/5 p-4 text-sm text-gray-400">
        No active positions.
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {positions.map((position) => (
        <motion.div
          key={`${position.symbol}-${position.opened_at || position.order_id}`}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="rounded-xl border border-white/10 bg-white/5 p-3"
        >
          <div className="flex items-center justify-between">
            <div className="text-sm font-semibold text-gray-100">
              {position.short_symbol} {position.side}
            </div>
            <div className={`text-sm font-semibold ${clsPnL(position.unrealized_pnl)}`}>
              {fmtUsd(position.unrealized_pnl)}
            </div>
          </div>
          <div className="mt-1 text-xs text-gray-300">
            Entry {fmtPrice(position.entry)} | SL {fmtPrice(position.sl)} | TP {fmtPrice(position.tp)}
          </div>
          <div className="mt-1 text-xs text-gray-400">
            Qty {Number(position.qty || 0).toFixed(3)} | Lev {Number(position.leverage || 0).toFixed(1)}x | Margin {fmtUsd(position.margin_used)}
          </div>
          <div className={`mt-1 text-xs ${clsPnL(position.unrealized_pnl_pct)}`}>
            Unrealized: {fmtPct(position.unrealized_pnl_pct)}
          </div>
        </motion.div>
      ))}
    </div>
  );
}

