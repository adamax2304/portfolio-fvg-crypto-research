import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { toast } from "sonner";

const PRIMARY_FIELDS = [
  "RISK_PER_TRADE",
  "MIN_GAP_PERCENT",
  "FVG_PERCENTILE_THRESHOLD",
  "SIGNAL_MIN_COMPOSITE_SCORE",
  "FVG_QUALITY_MIN_SCORE",
  "MAX_CONCURRENT_POSITIONS",
  "TOP_N_COINS",
  "MAX_LEVERAGE",
  "MIN_RISK_USD",
  "MAX_MARGIN_IN_USE_PCT",
  "PRE_ALERT_DISTANCE_PCT",
  "MID_ZONE_LOW",
  "MID_ZONE_HIGH",
  "SL_NEAREST_LEVEL_BUFFER_PCT",
  "TP_HVN_OFFSET_PCT",
];

function castValue(raw, original) {
  if (typeof original === "number") return Number(raw);
  if (typeof original === "boolean") return raw === "true" || raw === true;
  if (Array.isArray(original)) {
    try {
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : original;
    } catch {
      return original;
    }
  }
  return raw;
}

export default function BotSettingsPanel({ botSettings, onApply }) {
  const [draft, setDraft] = useState(botSettings || {});
  const [restartAfterApply, setRestartAfterApply] = useState(true);
  const [rawJson, setRawJson] = useState(JSON.stringify(botSettings || {}, null, 2));

  useEffect(() => {
    setDraft(botSettings || {});
    setRawJson(JSON.stringify(botSettings || {}, null, 2));
  }, [botSettings]);

  const fields = useMemo(() => {
    const keys = Object.keys(botSettings || {});
    const ordered = [...PRIMARY_FIELDS.filter((k) => keys.includes(k))];
    keys.forEach((key) => {
      if (!ordered.includes(key)) ordered.push(key);
    });
    return ordered;
  }, [botSettings]);

  const updateField = (key, value) => {
    const original = botSettings?.[key];
    const casted = castValue(value, original);
    setDraft((state) => ({ ...state, [key]: casted }));
  };

  const applySimple = async () => {
    await onApply(draft, restartAfterApply);
    setRawJson(JSON.stringify(draft, null, 2));
  };

  const applyJson = async () => {
    try {
      const parsed = JSON.parse(rawJson);
      await onApply(parsed, restartAfterApply);
      setDraft(parsed);
    } catch {
      toast.error("Invalid JSON in advanced editor");
    }
  };

  return (
    <div className="space-y-3">
      <div className="glass-panel rounded-2xl p-3">
        <div className="mb-2 text-sm font-semibold text-gray-100">Bot Settings</div>
        <div className="grid grid-cols-1 gap-2 md:grid-cols-2 xl:grid-cols-3">
          {fields.map((key) => (
            <label key={key} className="rounded-xl border border-white/10 bg-white/5 px-2 py-1 text-xs text-gray-300">
              {key}
              <input
                className="mt-1 w-full bg-transparent text-sm text-gray-100 outline-none"
                value={Array.isArray(draft[key]) ? JSON.stringify(draft[key]) : String(draft[key] ?? "")}
                onChange={(event) => updateField(key, event.target.value)}
              />
            </label>
          ))}
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-2 text-xs text-gray-300">
            <input
              type="checkbox"
              checked={restartAfterApply}
              onChange={(event) => setRestartAfterApply(event.target.checked)}
            />
            Apply & Restart Scan
          </label>
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.97 }}
            onClick={applySimple}
            className="rounded-xl border border-lime-400/40 bg-lime-400/12 px-3 py-2 text-xs text-lime-100"
          >
            Apply Bot Settings
          </motion.button>
        </div>
      </div>

      <div className="glass-panel rounded-2xl p-3">
        <div className="mb-2 text-sm font-semibold text-gray-100">Advanced JSON Editor (all bot params)</div>
        <textarea
          value={rawJson}
          onChange={(event) => setRawJson(event.target.value)}
          className="h-[260px] w-full rounded-xl border border-white/10 bg-black/30 p-3 font-mono text-xs text-gray-100 outline-none"
        />
        <div className="mt-2 flex justify-end">
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.97 }}
            onClick={applyJson}
            className="rounded-xl border border-white/20 bg-white/10 px-3 py-2 text-xs text-gray-100"
          >
            Apply JSON to Bot
          </motion.button>
        </div>
      </div>
    </div>
  );
}
