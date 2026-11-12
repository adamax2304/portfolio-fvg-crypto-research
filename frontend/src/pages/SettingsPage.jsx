import { useState } from "react";
import { useStore } from "../store/useStore";

export default function SettingsPage() {
  const { dashboard, botStart, botStop, updateSettings } = useStore();
  const [riskPct, setRiskPct] = useState(2.0);
  const [maxExposure, setMaxExposure] = useState(dashboard.max_exposure_pct || 8);
  const [minMargin, setMinMargin] = useState(5);

  const handleSave = () => {
    updateSettings({
      RISK_PER_TRADE: riskPct / 100,
      MAX_TOTAL_EXPOSURE_PCT: maxExposure,
      MIN_MARGIN_PER_TRADE_USD: minMargin,
    });
  };

  return (
    <div className="space-y-5 animate-fade-in max-w-2xl">
      <div>
        <h1 className="text-lg font-bold text-gray-100">Settings</h1>
        <p className="text-xs text-gray-500 mt-0.5">Bot control and risk parameters</p>
      </div>

      {/* Bot Control */}
      <div className="card p-5">
        <h2 className="text-sm font-semibold text-gray-200 mb-4">Bot Control</h2>
        <div className="flex items-center gap-3">
          <button onClick={botStart} className="btn-primary">Start Bot</button>
          <button onClick={botStop} className="btn-danger">Stop Bot</button>
        </div>
        <div className="mt-3 text-xs text-gray-500">
          Mode: {dashboard.simulation_mode ? (
            <span className="text-warn font-medium">Simulation (paper trading)</span>
          ) : (
            <span className="text-profit font-medium">LIVE Trading</span>
          )}
        </div>
      </div>

      {/* Risk Parameters */}
      <div className="card p-5">
        <h2 className="text-sm font-semibold text-gray-200 mb-4">Risk Management</h2>

        <div className="space-y-5">
          {/* Risk per trade */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-medium text-gray-300">Risk per Trade</label>
              <span className="text-xs font-mono text-accent">{riskPct.toFixed(1)}%</span>
            </div>
            <input
              type="range"
              min={0.5}
              max={5}
              step={0.1}
              value={riskPct}
              onChange={(e) => setRiskPct(parseFloat(e.target.value))}
              className="w-full h-1.5 rounded-full appearance-none cursor-pointer bg-dark-500 accent-accent"
            />
            <div className="flex justify-between text-[10px] text-gray-600 mt-1">
              <span>0.5%</span>
              <span>5.0%</span>
            </div>
          </div>

          {/* Max exposure */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-medium text-gray-300">Max Total Exposure</label>
              <span className="text-xs font-mono text-accent">{maxExposure.toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min={2}
              max={20}
              step={1}
              value={maxExposure}
              onChange={(e) => setMaxExposure(parseFloat(e.target.value))}
              className="w-full h-1.5 rounded-full appearance-none cursor-pointer bg-dark-500 accent-accent"
            />
            <div className="flex justify-between text-[10px] text-gray-600 mt-1">
              <span>2%</span>
              <span>20%</span>
            </div>
          </div>

          {/* Min margin */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-medium text-gray-300">Min Margin per Trade</label>
              <span className="text-xs font-mono text-accent">${minMargin}</span>
            </div>
            <input
              type="range"
              min={3}
              max={20}
              step={1}
              value={minMargin}
              onChange={(e) => setMinMargin(parseFloat(e.target.value))}
              className="w-full h-1.5 rounded-full appearance-none cursor-pointer bg-dark-500 accent-accent"
            />
            <div className="flex justify-between text-[10px] text-gray-600 mt-1">
              <span>$3</span>
              <span>$20</span>
            </div>
          </div>
        </div>

        <div className="mt-5 pt-4 border-t border-dark-700">
          <button onClick={handleSave} className="btn-primary">
            Save Settings
          </button>
        </div>
      </div>

      {/* Info */}
      <div className="card p-5">
        <h2 className="text-sm font-semibold text-gray-200 mb-3">Current Configuration</h2>
        <div className="grid grid-cols-2 gap-3 text-xs">
          <InfoRow label="Risk per trade" value={`${riskPct}% of total balance`} />
          <InfoRow label="Max exposure" value={`${maxExposure}% of deposit`} />
          <InfoRow label="Min margin" value={`$${minMargin} per trade`} />
          <InfoRow label="Max leverage" value="50x" />
          <InfoRow label="Max positions" value="6 total" />
          <InfoRow label="Max per symbol" value="1 position" />
          <InfoRow label="Daily loss stop" value="5% portfolio" />
          <InfoRow label="Order retries" value="1 attempt (no spam)" />
        </div>
      </div>
    </div>
  );
}

function InfoRow({ label, value }) {
  return (
    <div className="flex justify-between py-1.5 border-b border-dark-800/50">
      <span className="text-gray-500">{label}</span>
      <span className="text-gray-200 font-medium">{value}</span>
    </div>
  );
}
