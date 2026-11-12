import { motion } from "framer-motion";
import { clsPnL, fmtPct, fmtPrice, fmtTime } from "../lib/format";

export default function PendingSignalsPanel({ signals, onSelectSignal }) {
  if (!signals?.length) {
    return (
      <div className="rounded-2xl border border-white/10 bg-white/5 p-4 text-sm text-gray-400">
        No pending signals right now.
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 gap-2">
      {signals.map((signal, index) => (
        <motion.div
          key={`${signal.symbol}-${signal.timestamp}-${index}`}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          whileHover={{ scale: 1.015, y: -1 }}
          whileTap={{ scale: 0.98 }}
          onClick={() => onSelectSignal?.(signal)}
          className="cursor-pointer rounded-2xl border border-lime-400/20 bg-white/5 p-3 transition hover:border-lime-400/45 hover:bg-lime-400/10 hover:shadow-[0_0_22px_rgba(57,255,20,0.14)]"
        >
          <div className="flex items-center justify-between">
            <div className="max-w-[72%] truncate text-sm font-extrabold tracking-[0.01em] text-gray-100">
              {signal.short_symbol} {signal.signal_side}
            </div>
            <div className="text-xs text-gray-400">Score {Number(signal.score || 0).toFixed(1)}</div>
          </div>
          <div className="mt-1 text-xs font-semibold text-gray-300">
            Gap {fmtPct(signal.gap_pct)} | Entry {fmtPrice(signal.entry_price)} | RR {Number(signal.risk_reward || 0).toFixed(2)}
          </div>
          <div className="mt-1 text-xs text-gray-400">
            Zone {fmtPrice(signal.zone_low)} - {fmtPrice(signal.zone_high)} | formed {fmtTime(signal.formed_at)}
          </div>
          <div className={`mt-1 text-xs ${clsPnL(signal.risk_reward >= 1 ? 1 : -1)}`}>
            Status: {signal.entry_status || signal.status || "pending"}
          </div>
        </motion.div>
      ))}
    </div>
  );
}
