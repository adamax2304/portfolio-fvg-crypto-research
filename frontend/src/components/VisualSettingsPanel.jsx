import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import TogglePill from "./TogglePill";

export default function VisualSettingsPanel({
  visualSettings,
  chartToggles,
  glowIntensity,
  onApplyVisual,
  onToggleChange,
  onGlowChange,
}) {
  const [draft, setDraft] = useState({
    threshold: Number(visualSettings?.threshold ?? 0),
    min_gap_percent: Number(visualSettings?.min_gap_percent ?? 0),
    risk_percent: Number(visualSettings?.risk_percent ?? 2),
    bullish_color: visualSettings?.bullish_color || "#22c55e",
    bearish_color: visualSettings?.bearish_color || "#ef4444",
    entry_color: visualSettings?.entry_color || "#8cff66",
    sl_color: visualSettings?.sl_color || "#fb7185",
    tp_color: visualSettings?.tp_color || "#4ade80",
  });

  useEffect(() => {
    setDraft({
      threshold: Number(visualSettings?.threshold ?? 0),
      min_gap_percent: Number(visualSettings?.min_gap_percent ?? 0),
      risk_percent: Number(visualSettings?.risk_percent ?? 2),
      bullish_color: visualSettings?.bullish_color || "#22c55e",
      bearish_color: visualSettings?.bearish_color || "#ef4444",
      entry_color: visualSettings?.entry_color || "#8cff66",
      sl_color: visualSettings?.sl_color || "#fb7185",
      tp_color: visualSettings?.tp_color || "#4ade80",
    });
  }, [visualSettings]);

  return (
    <div className="space-y-3">
      <div className="glass-panel rounded-2xl p-3">
        <div className="mb-2 text-sm font-semibold text-gray-100">Visual Settings (dashboard only)</div>
        <div className="grid grid-cols-1 gap-2 md:grid-cols-2 xl:grid-cols-4">
          <label className="rounded-xl border border-white/10 bg-white/5 px-2 py-1 text-xs text-gray-300">
            Display min score
            <input
              type="number"
              value={draft.threshold}
              onChange={(event) => setDraft((state) => ({ ...state, threshold: Number(event.target.value || 0) }))}
              className="mt-1 w-full bg-transparent text-sm text-gray-100 outline-none"
            />
          </label>
          <label className="rounded-xl border border-white/10 bg-white/5 px-2 py-1 text-xs text-gray-300">
            Display min gap %
            <input
              type="number"
              step="0.01"
              value={draft.min_gap_percent}
              onChange={(event) => setDraft((state) => ({ ...state, min_gap_percent: Number(event.target.value || 0) }))}
              className="mt-1 w-full bg-transparent text-sm text-gray-100 outline-none"
            />
          </label>
          <label className="rounded-xl border border-white/10 bg-white/5 px-2 py-1 text-xs text-gray-300">
            Bullish color
            <input
              type="color"
              value={draft.bullish_color}
              onChange={(event) => setDraft((state) => ({ ...state, bullish_color: event.target.value }))}
              className="mt-1 h-8 w-full cursor-pointer bg-transparent"
            />
          </label>
          <label className="rounded-xl border border-white/10 bg-white/5 px-2 py-1 text-xs text-gray-300">
            Bearish color
            <input
              type="color"
              value={draft.bearish_color}
              onChange={(event) => setDraft((state) => ({ ...state, bearish_color: event.target.value }))}
              className="mt-1 h-8 w-full cursor-pointer bg-transparent"
            />
          </label>
          <label className="rounded-xl border border-white/10 bg-white/5 px-2 py-1 text-xs text-gray-300">
            Entry color
            <input
              type="color"
              value={draft.entry_color}
              onChange={(event) => setDraft((state) => ({ ...state, entry_color: event.target.value }))}
              className="mt-1 h-8 w-full cursor-pointer bg-transparent"
            />
          </label>
          <label className="rounded-xl border border-white/10 bg-white/5 px-2 py-1 text-xs text-gray-300">
            SL color
            <input
              type="color"
              value={draft.sl_color}
              onChange={(event) => setDraft((state) => ({ ...state, sl_color: event.target.value }))}
              className="mt-1 h-8 w-full cursor-pointer bg-transparent"
            />
          </label>
          <label className="rounded-xl border border-white/10 bg-white/5 px-2 py-1 text-xs text-gray-300">
            TP color
            <input
              type="color"
              value={draft.tp_color}
              onChange={(event) => setDraft((state) => ({ ...state, tp_color: event.target.value }))}
              className="mt-1 h-8 w-full cursor-pointer bg-transparent"
            />
          </label>
          <div className="rounded-xl border border-white/10 bg-white/5 px-2 py-2 text-xs text-gray-300">
            Glow intensity
            <div className="mt-2 flex gap-2">
              <button
                onClick={() => onGlowChange("low")}
                className={`rounded-lg px-2 py-1 ${glowIntensity === "low" ? "bg-lime-400/25 text-lime-100" : "bg-white/10 text-gray-300"}`}
              >
                Low
              </button>
              <button
                onClick={() => onGlowChange("high")}
                className={`rounded-lg px-2 py-1 ${glowIntensity === "high" ? "bg-lime-400/25 text-lime-100" : "bg-white/10 text-gray-300"}`}
              >
                High
              </button>
            </div>
          </div>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-2">
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
        </div>

        <div className="mt-3 flex justify-end">
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.97 }}
            onClick={() =>
              onApplyVisual({
                ...draft,
                show_zones: chartToggles.showZones,
                show_levels: chartToggles.showLevels,
                show_labels: chartToggles.showLabels,
                show_mitigated: chartToggles.showMitigated,
                glow_intensity: glowIntensity,
              })
            }
            className="rounded-xl border border-lime-400/40 bg-lime-400/12 px-3 py-2 text-xs text-lime-100"
          >
            Apply Visual Settings
          </motion.button>
        </div>
      </div>
    </div>
  );
}
