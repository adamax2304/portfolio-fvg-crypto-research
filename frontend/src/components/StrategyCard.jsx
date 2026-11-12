import { useStore } from "../store/useStore";

export default function StrategyCard({ strategy }) {
  const toggleStrategy = useStore((s) => s.toggleStrategy);
  const bt = strategy.backtest || {};

  return (
    <div className="card p-5 card-glow animate-slide-up">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className={`inline-block w-2 h-2 rounded-full ${
              strategy.enabled ? "bg-profit pulse-dot" : "bg-gray-500"
            }`} />
            <h3 className="text-sm font-bold text-gray-100 truncate">{strategy.name}</h3>
          </div>
          <p className="text-xs text-gray-500 truncate">{strategy.description}</p>
        </div>
        {/* Toggle */}
        <button
          onClick={() => toggleStrategy(strategy.id)}
          className={`toggle-track flex-shrink-0 ${strategy.enabled ? "active" : ""}`}
        >
          <div className="toggle-thumb" />
        </button>
      </div>

      {/* TF Info */}
      <div className="flex gap-2 mb-4">
        <span className="px-2 py-1 rounded-md bg-dark-600/60 text-[11px] font-mono text-gray-300">
          Signal: {strategy.signal_tf}
        </span>
        <span className="px-2 py-1 rounded-md bg-dark-600/60 text-[11px] font-mono text-gray-300">
          Entry: {strategy.entry_tf}
        </span>
        <span className="px-2 py-1 rounded-md bg-dark-600/60 text-[11px] font-mono text-gray-300">
          {strategy.entry_mode === "pullback_ema21" ? "Pullback" : "Next Bar"}
        </span>
      </div>

      {/* Backtest Stats Grid */}
      <div className="grid grid-cols-4 gap-3 mb-3">
        <StatCell label="PF" value={bt.pf?.toFixed(2) || "-"} color="text-accent" />
        <StatCell label="Win Rate" value={bt.wr_pct ? bt.wr_pct.toFixed(0) + "%" : "-"} />
        <StatCell label="PnL" value={bt.pnl_pct ? "+" + bt.pnl_pct.toFixed(0) + "%" : "-"} color="text-profit" />
        <StatCell label="Max DD" value={bt.dd_pct ? bt.dd_pct.toFixed(1) + "%" : "-"} color="text-loss" />
      </div>

      {/* SL/TP Info */}
      <div className="flex items-center justify-between text-[11px] text-gray-500 pt-3 border-t border-dark-700">
        <span>SL: ATR x{strategy.sl_atr_mult}</span>
        <span>TP: {strategy.tp_r}R</span>
        <span>WF Test: {bt.wf_test_pf?.toFixed(2) || "-"}</span>
      </div>

      {/* Live stats */}
      <div className="flex items-center gap-4 mt-3 pt-3 border-t border-dark-700">
        <div className="text-xs">
          <span className="text-gray-500">Today: </span>
          <span className={strategy.pnl_today > 0 ? "text-profit" : strategy.pnl_today < 0 ? "text-loss" : "text-gray-300"}>
            {strategy.pnl_today ? (strategy.pnl_today > 0 ? "+" : "") + strategy.pnl_today.toFixed(2) + "%" : "0"}
          </span>
        </div>
        <div className="text-xs">
          <span className="text-gray-500">Trades: </span>
          <span className="text-gray-300">{strategy.trades_today || 0}</span>
        </div>
        <div className="text-xs">
          <span className="text-gray-500">Open: </span>
          <span className="text-gray-300">{strategy.active_positions || 0}</span>
        </div>
      </div>
    </div>
  );
}

function StatCell({ label, value, color }) {
  return (
    <div>
      <div className="text-[10px] text-gray-500 mb-0.5">{label}</div>
      <div className={`text-sm font-bold ${color || "text-gray-200"}`}>{value}</div>
    </div>
  );
}
