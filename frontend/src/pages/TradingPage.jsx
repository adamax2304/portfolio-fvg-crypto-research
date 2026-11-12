import ActivePositionsPanel from "../components/ActivePositionsPanel";
import PendingSignalsPanel from "../components/PendingSignalsPanel";
import TradeHistoryTable from "../components/TradeHistoryTable";

export default function TradingPage({ positions, pendingSignals, tradeHistory }) {
  return (
    <div className="grid h-full grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <div className="space-y-3">
        <div className="glass-panel rounded-2xl p-3">
          <div className="mb-2 text-sm font-semibold text-gray-100">Current Positions</div>
          <ActivePositionsPanel positions={positions} />
        </div>
        <div className="glass-panel rounded-2xl p-3">
          <div className="mb-2 text-sm font-semibold text-gray-100">Pending Signals (sorted by score)</div>
          <PendingSignalsPanel signals={pendingSignals} />
        </div>
      </div>
      <TradeHistoryTable rows={tradeHistory} />
    </div>
  );
}

