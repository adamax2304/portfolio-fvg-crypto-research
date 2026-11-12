import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { fmtPct, fmtUsd } from "../lib/format";

export default function AnalyticsPanel({ analytics }) {
  const equity = (analytics?.equity_curve || []).map((point, index) => ({
    index,
    balance: Number(point.balance || point.equity || 0),
    timestamp: point.timestamp || point.time || index,
  }));

  const winLoss = [
    { name: "Win", value: Number(analytics?.wins || 0), color: "#16e6a5" },
    { name: "Loss", value: Number(analytics?.losses || 0), color: "#ff5a74" },
  ];

  const heatmapData = (analytics?.heatmap || []).slice(-30).map((row) => ({
    date: row.date,
    pnl_usd: Number(row.pnl_usd || 0),
  }));

  return (
    <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
      <div className="glass-panel rounded-2xl p-3">
        <div className="mb-2 text-sm font-semibold text-gray-100">
          Equity Curve | Total {fmtUsd(analytics?.total_pnl)}
        </div>
        <div className="h-[280px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={equity}>
              <defs>
                <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#facc15" stopOpacity={0.55} />
                  <stop offset="95%" stopColor="#facc15" stopOpacity={0.03} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="rgba(148,163,184,0.12)" />
              <XAxis dataKey="index" tick={{ fill: "#9ca3af", fontSize: 11 }} />
              <YAxis tick={{ fill: "#9ca3af", fontSize: 11 }} />
              <Tooltip />
              <Area
                type="monotone"
                dataKey="balance"
                stroke="#facc15"
                fill="url(#equityGrad)"
                strokeWidth={2}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="glass-panel rounded-2xl p-3">
        <div className="mb-2 text-sm font-semibold text-gray-100">
          Win vs Loss | WR {fmtPct(analytics?.win_rate)}
        </div>
        <div className="h-[280px]">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie data={winLoss} dataKey="value" nameKey="name" innerRadius={64} outerRadius={96}>
                {winLoss.map((entry) => (
                  <Cell key={entry.name} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="glass-panel rounded-2xl p-3 xl:col-span-2">
        <div className="mb-2 text-sm font-semibold text-gray-100">
          Performance Heatmap (last 30 trading days)
        </div>
        <div className="h-[220px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={heatmapData}>
              <CartesianGrid stroke="rgba(148,163,184,0.12)" />
              <XAxis dataKey="date" tick={{ fill: "#9ca3af", fontSize: 11 }} />
              <YAxis tick={{ fill: "#9ca3af", fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="pnl_usd" radius={[4, 4, 0, 0]}>
                {heatmapData.map((row, idx) => (
                  <Cell key={`h-${idx}`} fill={row.pnl_usd >= 0 ? "#22c55e" : "#ef4444"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

