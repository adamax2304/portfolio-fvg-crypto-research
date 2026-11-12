import {
  LayoutDashboard, Zap, Crosshair, History,
  Settings, FileText, ChevronLeft, ChevronRight, Bot,
} from "lucide-react";
import { useStore } from "../store/useStore";

const NAV = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "strategies", label: "Strategies", icon: Zap },
  { id: "signals", label: "Signals", icon: Crosshair },
  { id: "trades", label: "Trades", icon: History },
  { id: "settings", label: "Settings", icon: Settings },
  { id: "logs", label: "Logs", icon: FileText },
];

export default function Sidebar() {
  const { activeTab, setTab, sidebarOpen, toggleSidebar, dashboard } = useStore();

  return (
    <aside
      className={`flex flex-col border-r border-dark-700 bg-dark-900 transition-all duration-200 ${
        sidebarOpen ? "w-[220px]" : "w-[64px]"
      }`}
    >
      {/* Logo area */}
      <div className="flex items-center gap-3 px-4 py-4 border-b border-dark-700">
        <div className="w-8 h-8 rounded-lg bg-accent/15 flex items-center justify-center flex-shrink-0">
          <Bot size={18} className="text-accent" />
        </div>
        {sidebarOpen && (
          <div className="min-w-0">
            <div className="text-sm font-bold text-gray-100 truncate">MTF Bot</div>
            <div className="text-[10px] text-gray-500 font-medium">v10 Multi-TF</div>
          </div>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-3 px-2 space-y-0.5">
        {NAV.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={`nav-item w-full ${activeTab === id ? "active" : ""}`}
            title={!sidebarOpen ? label : undefined}
          >
            <Icon size={18} />
            {sidebarOpen && <span>{label}</span>}
          </button>
        ))}
      </nav>

      {/* Status indicator */}
      {sidebarOpen && (
        <div className="px-4 pb-3">
          <div className="card p-3 space-y-2">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${
                dashboard.simulation_mode ? "bg-warn pulse-dot" : "bg-profit pulse-dot"
              }`} />
              <span className="text-xs text-gray-400">
                {dashboard.simulation_mode ? "Paper Mode" : "Live Trading"}
              </span>
            </div>
            <div className="text-xs text-gray-500">
              {dashboard.active_positions} open / {dashboard.total_trades} total
            </div>
          </div>
        </div>
      )}

      {/* Collapse toggle */}
      <button
        onClick={toggleSidebar}
        className="flex items-center justify-center py-3 border-t border-dark-700 text-gray-500 hover:text-gray-300 transition-colors"
      >
        {sidebarOpen ? <ChevronLeft size={16} /> : <ChevronRight size={16} />}
      </button>
    </aside>
  );
}
