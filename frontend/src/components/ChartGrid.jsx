import ChartCard from "./ChartCard";

export default function ChartGrid({
  selectedSymbols,
  chartDataBySymbol,
  visual,
  toggles,
  glowIntensity,
}) {
  if (!selectedSymbols.length) {
    return (
      <div className="glass-panel flex h-[420px] items-center justify-center rounded-3xl text-sm font-semibold text-gray-400">
        Select symbols from the left sidebar to open charts.
      </div>
    );
  }

  const multi = selectedSymbols.length > 1;
  return (
    <div className={multi ? "grid grid-cols-1 gap-3 xl:grid-cols-2" : "grid grid-cols-1 gap-3"}>
      {selectedSymbols.map((symbol) => (
        <ChartCard
          key={symbol}
          symbol={symbol}
          chartData={chartDataBySymbol[symbol]}
          visual={visual}
          toggles={toggles}
          glowIntensity={glowIntensity}
        />
      ))}
    </div>
  );
}
