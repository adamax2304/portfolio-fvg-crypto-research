import ChartGrid from "../components/ChartGrid";
import TogglePill from "../components/TogglePill";

export default function ChartsPage({
  selectedSymbols,
  chartDataBySymbol,
  visual,
  chartToggles,
  glowIntensity,
  timeframe,
  historyDays,
  onTimeframeChange,
  onHistoryDaysChange,
  onToggleChange,
  focusMode,
  onFocusModeChange,
}) {
  return (
    <div className="space-y-3">
      <div className="glass-panel rounded-2xl p-3">
        <div className="flex flex-wrap items-center gap-2">
          <div className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-xs text-gray-300">
            Timeframe
          </div>
          <button
            onClick={() => onTimeframeChange("5m")}
            className={`rounded-lg px-3 py-2 text-xs ${timeframe === "5m" ? "bg-lime-400/20 text-lime-100" : "bg-white/5 text-gray-300"}`}
          >
            5m
          </button>
          <button
            onClick={() => onTimeframeChange("15m")}
            className={`rounded-lg px-3 py-2 text-xs ${timeframe === "15m" ? "bg-lime-400/20 text-lime-100" : "bg-white/5 text-gray-300"}`}
          >
            15m
          </button>
          <div className="ml-2 rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-xs text-gray-300">
            History
          </div>
          {[5, 10, 20, 30, 60].map((days) => (
            <button
              key={days}
              onClick={() => onHistoryDaysChange(days)}
              className={`rounded-lg px-3 py-2 text-xs ${
                Number(historyDays) === days
                  ? "bg-lime-400/25 text-lime-100"
                  : "bg-white/5 text-gray-300"
              }`}
            >
              {days}d
            </button>
          ))}
          <button
            onClick={() => onHistoryDaysChange("max")}
            className={`rounded-lg px-3 py-2 text-xs ${
              historyDays === "max"
                ? "bg-lime-400/25 text-lime-100"
                : "bg-white/5 text-gray-300"
            }`}
          >
            Max
          </button>
          <TogglePill
            label="Show Zones"
            checked={chartToggles.showZones}
            onChange={(value) => onToggleChange("showZones", value)}
          />
          <TogglePill
            label="Show SL-TP"
            checked={chartToggles.showLevels}
            onChange={(value) => onToggleChange("showLevels", value)}
          />
          <TogglePill
            label="Show Labels"
            checked={chartToggles.showLabels}
            onChange={(value) => onToggleChange("showLabels", value)}
          />
          <TogglePill
            label="Show Mitigated"
            checked={chartToggles.showMitigated}
            onChange={(value) => onToggleChange("showMitigated", value)}
          />
          <TogglePill
            label="Focus Mode"
            checked={focusMode}
            onChange={onFocusModeChange}
          />
        </div>
      </div>

      <ChartGrid
        selectedSymbols={selectedSymbols}
        chartDataBySymbol={chartDataBySymbol}
        visual={visual}
        toggles={chartToggles}
        glowIntensity={glowIntensity}
      />
    </div>
  );
}
