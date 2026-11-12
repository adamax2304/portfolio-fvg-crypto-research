import StrategyCard from "../components/StrategyCard";
import { useStore } from "../store/useStore";

export default function StrategiesPage() {
  const strategies = useStore((s) => s.strategies);

  return (
    <div className="space-y-5 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-lg font-bold text-gray-100">Strategies</h1>
          <p className="text-xs text-gray-500 mt-0.5">
            Multi-TF strategies — v10 optimized, walk-forward confirmed
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">
            {strategies.filter((s) => s.enabled).length}/{strategies.length} active
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {strategies.map((s) => (
          <StrategyCard key={s.id} strategy={s} />
        ))}
      </div>

      {strategies.length === 0 && (
        <div className="card p-8 text-center text-gray-500 text-sm">
          No strategies configured. Check backend connection.
        </div>
      )}

      {/* Strategy Info */}
      <div className="card p-5">
        <h3 className="text-sm font-semibold text-gray-200 mb-3">How MTF Strategies Work</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-gray-400">
          <div className="space-y-1">
            <div className="font-medium text-accent">1. Signal Detection</div>
            <p>Higher timeframe (1h) detects BB Squeeze or KC Squeeze patterns with ADX trend confirmation.</p>
          </div>
          <div className="space-y-1">
            <div className="font-medium text-accent">2. Entry Execution</div>
            <p>Lower timeframe (5m/15m) provides precise entry via pullback to EMA21 or next-bar confirmation.</p>
          </div>
          <div className="space-y-1">
            <div className="font-medium text-accent">3. Risk Management</div>
            <p>ATR-based stop loss (1.5-2.0x) with fixed R-multiple take profit. Max 8% total deposit exposure.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
