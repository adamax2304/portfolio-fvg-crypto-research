import { motion } from "framer-motion";
import {
  Bell,
  Bot,
  CandlestickChart,
  LayoutDashboard,
  RefreshCcw,
  SlidersHorizontal,
  TableProperties,
  TerminalSquare,
  Trash2,
  Waypoints,
} from "lucide-react";

const TABS = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "trading", label: "Trading Panel", icon: TableProperties },
  { id: "charts", label: "Advanced Charts", icon: CandlestickChart },
  { id: "analytics", label: "Analytics", icon: Waypoints },
  { id: "logs", label: "Bot Log", icon: TerminalSquare },
  { id: "botSettings", label: "Bot Settings", icon: Bot },
  { id: "visualSettings", label: "Visual Settings", icon: SlidersHorizontal },
];

export default function TopBar({
  activeTab,
  onTabChange,
  onRefresh,
  onClearCharts,
  botRunning,
}) {
  return (
    <div className="glass-panel rounded-3xl p-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <div className="rounded-2xl bg-white/5 px-3 py-2 text-sm font-extrabold tracking-wide text-gray-100">
            FVG Terminal v2.6
          </div>
          <div className="flex items-center gap-2 rounded-2xl border border-lime-400/35 bg-lime-400/10 px-3 py-2 text-xs font-semibold text-gray-200">
            <span
              className={`h-2.5 w-2.5 rounded-full ${
                botRunning ? "animate-pulse bg-emerald-400" : "bg-rose-400"
              }`}
            />
            {botRunning ? "Bot Running" : "Bot Stopped"}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <motion.button
            whileHover={{ scale: 1.02, y: -1 }}
            whileTap={{ scale: 0.95 }}
            onClick={onClearCharts}
            className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs font-semibold text-gray-200 hover:border-lime-300/50 hover:bg-lime-400/12"
          >
            <Trash2 size={14} />
            Clear All Charts
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.02, y: -1 }}
            whileTap={{ scale: 0.95 }}
            onClick={onRefresh}
            className="inline-flex items-center gap-2 rounded-xl border border-lime-400/55 bg-lime-400/16 px-3 py-2 text-xs font-semibold text-lime-100 shadow-[0_0_20px_rgba(57,255,20,0.14)]"
          >
            <RefreshCcw size={14} />
            Refresh All
          </motion.button>
          <button className="rounded-xl border border-white/10 bg-white/5 p-2 text-gray-300 hover:border-lime-300/50 hover:bg-lime-400/12">
            <Bell size={14} />
          </button>
        </div>
      </div>

      <div className="mt-3 flex flex-wrap gap-2">
        {TABS.map((tab) => {
          const Icon = tab.icon;
          const active = activeTab === tab.id;
          return (
            <motion.button
              key={tab.id}
              whileHover={{ scale: 1.02, y: -1 }}
              whileTap={{ scale: 0.96 }}
              onClick={() => onTabChange(tab.id)}
              className={`inline-flex items-center gap-2 rounded-2xl px-3 py-2 text-xs font-semibold transition ${
                active
                  ? "bg-lime-400/20 text-lime-100 shadow-[0_0_22px_rgba(57,255,20,0.16)]"
                  : "bg-white/5 text-gray-300 hover:bg-white/10"
              }`}
            >
              <Icon size={14} />
              {tab.label}
            </motion.button>
          );
        })}
      </div>
    </div>
  );
}
