import { Wallet, TrendingUp, Trophy, Activity } from "lucide-react";
import MetricCard from "../components/MetricCard";
import EquityCurve from "../components/EquityCurve";
import { useStore } from "../store/useStore";
import { fmtUsd, fmtPct, fmtPrice, fmtTime, pnlColor } from "../lib/format";

export default function DashboardPage() {
  const { dashboard, equityCurve, strategies, trades } = useStore();
  const d = dashboard;

  return (
    <div className="space-y-5 animate-fade-in">
      {/* Metric Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="Balance"
          value={fmtUsd(d.balance)}
          icon={Wallet}
        />
        <MetricCard
          label="Daily P&L"
          value={fmtPct(d.daily_pnl ? (d.daily_pnl / Math.max(d.balance, 1)) * 100 : 0)}
          sub={fmtUsd(d.daily_pnl)}
          color={d.daily_pnl >= 0 ? "text-profit" : "text-loss"}
          icon={TrendingUp}
        />
        <MetricCard
          label="Win Rate"
          value={d.win_rate.toFixed(1) + "%"}
          sub={`${d.wins}W / ${d.losses}L`}
          color="text-accent"
          icon={Trophy}
        />
        <MetricCard
          label="Total Trades"
          value={d.total_trades}
          sub={`${d.active_positions} open positions`}
          icon={Activity}
        />
      </div>

      {/* Equity Curve + Strategies Summary */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 card p-4">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-semibold text-gray-200">Equity Curve</h2>
            <span className="text-xs text-gray-500">
              Total PnL: <span className={pnlColor(d.total_pnl)}>{fmtUsd(d.total_pnl)}</span>
            </span>
          </div>
          <div className="h-[260px]">
            <EquityCurve data={equityCurve} />
          </div>
        </div>

        <div className="card p-4">
          <h2 className="text-sm font-semibold text-gray-200 mb-3">Strategies</h2>
          <div className="space-y-3">
            {strategies.map((s) => (
              <div key={s.id} className="flex items-center justify-between py-2 border-b border-dark-700 last:border-0">
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${s.enabled ? "bg-profit" : "bg-gray-600"}`} />
                  <div>
                    <div className="text-xs font-medium text-gray-200">{s.name}</div>
                    <div className="text-[10px] text-gray-500">{s.signal_tf} &rarr; {s.entry_tf}</div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-xs font-mono text-accent">PF {s.backtest?.pf?.toFixed(1)}</div>
                  <div className="text-[10px] text-gray-500">WF {s.backtest?.wf_test_pf?.toFixed(1)}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Active Positions */}
      <div className="card p-4">
        <h2 className="text-sm font-semibold text-gray-200 mb-3">
          Active Positions
          <span className="ml-2 text-xs text-gray-500 font-normal">({d.active_positions})</span>
        </h2>
        {d.positions && d.positions.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-gray-500 border-b border-dark-700">
                  <th className="text-left py-2 font-medium">Symbol</th>
                  <th className="text-left py-2 font-medium">Side</th>
                  <th className="text-right py-2 font-medium">Entry</th>
                  <th className="text-right py-2 font-medium">Current</th>
                  <th className="text-right py-2 font-medium">Size</th>
                  <th className="text-right py-2 font-medium">PnL</th>
                </tr>
              </thead>
              <tbody>
                {d.positions.map((p, i) => (
                  <tr key={i} className="border-b border-dark-800 hover:bg-dark-700/30">
                    <td className="py-2 font-medium text-gray-200">{p.symbol?.replace("/USDT:USDT", "") || "-"}</td>
                    <td className={`py-2 font-semibold ${p.side === "LONG" ? "text-profit" : "text-loss"}`}>
                      {p.side}
                    </td>
                    <td className="py-2 text-right font-mono text-gray-300">{fmtPrice(p.entry)}</td>
                    <td className="py-2 text-right font-mono text-gray-300">{fmtPrice(p.mark_price)}</td>
                    <td className="py-2 text-right text-gray-400">{p.qty}</td>
                    <td className={`py-2 text-right font-semibold ${pnlColor(p.unrealized_pnl)}`}>
                      {fmtUsd(p.unrealized_pnl)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-sm text-gray-500 text-center py-6">No open positions</div>
        )}
      </div>

      {/* Recent Trades */}
      <div className="card p-4">
        <h2 className="text-sm font-semibold text-gray-200 mb-3">Recent Trades</h2>
        {trades.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-gray-500 border-b border-dark-700">
                  <th className="text-left py-2 font-medium">Time</th>
                  <th className="text-left py-2 font-medium">Symbol</th>
                  <th className="text-left py-2 font-medium">Side</th>
                  <th className="text-right py-2 font-medium">Entry</th>
                  <th className="text-right py-2 font-medium">Exit</th>
                  <th className="text-right py-2 font-medium">PnL</th>
                  <th className="text-left py-2 font-medium">Reason</th>
                </tr>
              </thead>
              <tbody>
                {trades.slice(0, 10).map((t, i) => (
                  <tr key={i} className="border-b border-dark-800 hover:bg-dark-700/30">
                    <td className="py-2 text-gray-400">{fmtTime(t.exit_time || t.entry_time)}</td>
                    <td className="py-2 font-medium text-gray-200">
                      {(t.symbol || "").replace("/USDT:USDT", "")}
                    </td>
                    <td className={`py-2 font-semibold ${(t.direction || t.side) === "LONG" ? "text-profit" : "text-loss"}`}>
                      {t.direction || t.side}
                    </td>
                    <td className="py-2 text-right font-mono text-gray-300">{fmtPrice(t.entry_price)}</td>
                    <td className="py-2 text-right font-mono text-gray-300">{fmtPrice(t.exit_price)}</td>
                    <td className={`py-2 text-right font-semibold ${pnlColor(t.pnl_usd)}`}>
                      {fmtUsd(t.pnl_usd)}
                    </td>
                    <td className="py-2 text-gray-400">{t.exit_reason || "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-sm text-gray-500 text-center py-6">No trades yet</div>
        )}
      </div>
    </div>
  );
}
