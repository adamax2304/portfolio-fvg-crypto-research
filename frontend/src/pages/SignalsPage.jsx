import { useEffect, useState } from "react";
import MiniChart from "../components/MiniChart";
import { useStore } from "../store/useStore";
import { fmtPrice, pnlColor } from "../lib/format";

export default function SignalsPage() {
  const { signals, fetchChart, chartData } = useStore();
  const [selectedSignal, setSelectedSignal] = useState(null);

  // Load chart for selected signal
  useEffect(() => {
    if (selectedSignal?.symbol) {
      const tf = selectedSignal.entry_tf || "15m";
      fetchChart(selectedSignal.symbol, tf);
    }
  }, [selectedSignal]);

  const activeSignal = selectedSignal || (signals.length > 0 ? signals[0] : null);
  const chartKey = activeSignal?.symbol;
  const candles = chartData[chartKey]?.candles || [];

  return (
    <div className="space-y-5 animate-fade-in">
      <div>
        <h1 className="text-lg font-bold text-gray-100">Potential Entries</h1>
        <p className="text-xs text-gray-500 mt-0.5">
          Signals detected by active strategies. Click to view chart with entry/SL/TP levels.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Signal list */}
        <div className="space-y-2">
          {signals.length > 0 ? (
            signals.map((sig, i) => {
              const sym = (sig.symbol || sig.short_symbol || "").replace("/USDT:USDT", "");
              const isSelected = activeSignal === sig;
              const isLong = (sig.signal_side || sig.direction || "").toUpperCase() === "LONG";

              return (
                <button
                  key={i}
                  onClick={() => setSelectedSignal(sig)}
                  className={`w-full text-left card p-3 transition-all ${
                    isSelected ? "border-accent/40 shadow-glow" : ""
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-semibold text-gray-100">{sym}</span>
                    <span className={`text-xs font-bold px-2 py-0.5 rounded ${
                      isLong ? "bg-profit/15 text-profit" : "bg-loss/15 text-loss"
                    }`}>
                      {isLong ? "LONG" : "SHORT"}
                    </span>
                  </div>

                  <div className="grid grid-cols-3 gap-2 text-[11px]">
                    <div>
                      <span className="text-gray-500">Entry</span>
                      <div className="font-mono text-accent">{fmtPrice(sig.entry_price)}</div>
                    </div>
                    <div>
                      <span className="text-gray-500">SL</span>
                      <div className="font-mono text-loss">{fmtPrice(sig.sl_price || sig.sl)}</div>
                    </div>
                    <div>
                      <span className="text-gray-500">TP</span>
                      <div className="font-mono text-profit">{fmtPrice(sig.tp_price || sig.tp)}</div>
                    </div>
                  </div>

                  <div className="flex items-center gap-3 mt-2 text-[10px] text-gray-500">
                    {sig.strategy && <span className="px-1.5 py-0.5 rounded bg-dark-600/50">{sig.strategy}</span>}
                    {sig.risk_reward && <span>RR: {Number(sig.risk_reward).toFixed(1)}</span>}
                    {sig.score && <span>Score: {Number(sig.score).toFixed(0)}</span>}
                  </div>
                </button>
              );
            })
          ) : (
            <div className="card p-6 text-center text-gray-500 text-sm">
              No active signals detected.
              <div className="text-xs text-gray-600 mt-1">Signals appear when strategies detect setups.</div>
            </div>
          )}
        </div>

        {/* Chart area */}
        <div className="lg:col-span-2 card p-4">
          {activeSignal ? (
            <>
              <div className="flex items-center justify-between mb-3">
                <div>
                  <span className="text-sm font-bold text-gray-100">
                    {(activeSignal.symbol || "").replace("/USDT:USDT", "")}
                  </span>
                  <span className="ml-2 text-xs text-gray-500">{activeSignal.entry_tf || "15m"}</span>
                </div>
                <div className="flex items-center gap-4 text-xs">
                  <div>
                    <span className="text-gray-500">Entry: </span>
                    <span className="font-mono text-accent">{fmtPrice(activeSignal.entry_price)}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">SL: </span>
                    <span className="font-mono text-loss">{fmtPrice(activeSignal.sl_price || activeSignal.sl)}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">TP: </span>
                    <span className="font-mono text-profit">{fmtPrice(activeSignal.tp_price || activeSignal.tp)}</span>
                  </div>
                </div>
              </div>
              <div className="h-[400px]">
                <MiniChart symbol={chartKey} candles={candles} signal={activeSignal} />
              </div>
            </>
          ) : (
            <div className="flex items-center justify-center h-[400px] text-gray-500 text-sm">
              Select a signal to view chart
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
