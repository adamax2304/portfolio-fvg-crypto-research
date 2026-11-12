import { useEffect } from "react";
import { useStore } from "./store/useStore";
import Sidebar from "./components/Sidebar";
import DashboardPage from "./pages/DashboardPage";
import StrategiesPage from "./pages/StrategiesPage";
import SignalsPage from "./pages/SignalsPage";
import TradesPage from "./pages/TradesPage";
import SettingsPage from "./pages/SettingsPage";
import LogsPage from "./pages/LogsPage";

const PAGES = {
  dashboard: DashboardPage,
  strategies: StrategiesPage,
  signals: SignalsPage,
  trades: TradesPage,
  settings: SettingsPage,
  logs: LogsPage,
};

export default function App() {
  const { activeTab, isLoading, error, bootstrap, disconnectWs } = useStore();

  useEffect(() => {
    bootstrap();
    return () => disconnectWs();
  }, []);

  // Auto-refresh every 30s
  useEffect(() => {
    const iv = setInterval(() => useStore.getState().refresh(), 30000);
    return () => clearInterval(iv);
  }, []);

  const Page = PAGES[activeTab] || DashboardPage;

  return (
    <div className="h-full w-full flex bg-dark-950">
      <Sidebar />
      <main className="flex-1 min-h-0 flex flex-col overflow-hidden">
        {/* Top status bar */}
        <TopStatusBar />
        {/* Main content */}
        <div className="flex-1 min-h-0 overflow-y-auto scroll-y p-5">
          {error && (
            <div className="mb-4 rounded-lg bg-loss/10 border border-loss/20 px-4 py-3 text-sm text-loss">
              {error}
            </div>
          )}
          {isLoading ? (
            <div className="flex items-center justify-center h-64 text-gray-400">
              <div className="flex items-center gap-3">
                <div className="w-5 h-5 border-2 border-accent/30 border-t-accent rounded-full animate-spin" />
                Loading dashboard...
              </div>
            </div>
          ) : (
            <Page />
          )}
        </div>
      </main>
    </div>
  );
}

function TopStatusBar() {
  const { dashboard } = useStore();
  const sim = dashboard.simulation_mode;
  return (
    <div className="flex items-center justify-between px-5 py-2.5 border-b border-dark-700 bg-dark-900/80 backdrop-blur-sm">
      <div className="flex items-center gap-4">
        <span className="text-xs font-medium text-gray-400">
          Mode: {sim ? (
            <span className="text-warn font-semibold">SIMULATION</span>
          ) : (
            <span className="text-profit font-semibold">LIVE</span>
          )}
        </span>
        <span className="text-xs text-gray-500">|</span>
        <span className="text-xs text-gray-400">
          Positions: <span className="text-gray-200 font-medium">{dashboard.active_positions}</span>
        </span>
        <span className="text-xs text-gray-500">|</span>
        <span className="text-xs text-gray-400">
          Max exposure: <span className="text-gray-200 font-medium">{dashboard.max_exposure_pct}%</span>
        </span>
      </div>
      <div className="flex items-center gap-3">
        <span className="text-xs text-gray-400">
          Balance: <span className="text-gray-100 font-semibold">${Number(dashboard.balance).toFixed(2)}</span>
        </span>
      </div>
    </div>
  );
}
