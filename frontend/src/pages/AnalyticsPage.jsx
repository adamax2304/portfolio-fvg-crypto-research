import AnalyticsPanel from "../components/AnalyticsPanel";
import TradeHistoryTable from "../components/TradeHistoryTable";

export default function AnalyticsPage({ analytics, tradeHistory }) {
  return (
    <div className="space-y-3">
      <AnalyticsPanel analytics={analytics} />
      <TradeHistoryTable rows={tradeHistory} />
    </div>
  );
}

