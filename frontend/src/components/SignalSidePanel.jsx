import PendingSignalsPanel from "./PendingSignalsPanel";
import ActivePositionsPanel from "./ActivePositionsPanel";
import LogsPanel from "./LogsPanel";

const TABS = [
  { id: "pending", label: "Pending Signals" },
  { id: "positions", label: "Active Positions" },
  { id: "logs", label: "Bot Log" },
];

export default function SignalSidePanel({
  activeTab,
  onTabChange,
  pendingSignals,
  onSelectSignal,
  activePositions,
  logs,
}) {
  return (
    <div className="glass-panel flex h-full min-h-0 flex-col rounded-3xl p-3">
      <div className="mb-2 flex gap-2">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={`rounded-xl px-3 py-2 text-xs font-semibold ${
              activeTab === tab.id
                ? "bg-lime-400/22 text-lime-100 shadow-[0_0_22px_rgba(57,255,20,0.14)]"
                : "bg-white/5 text-gray-300 hover:bg-white/10"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="scroll-y min-h-0 flex-1 pr-1">
        {activeTab === "pending" ? (
          <PendingSignalsPanel signals={pendingSignals} onSelectSignal={onSelectSignal} />
        ) : null}
        {activeTab === "positions" ? (
          <ActivePositionsPanel positions={activePositions} />
        ) : null}
        {activeTab === "logs" ? <LogsPanel lines={logs} /> : null}
      </div>
    </div>
  );
}
