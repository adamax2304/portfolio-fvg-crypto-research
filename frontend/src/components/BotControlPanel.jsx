import { useState } from "react";
import { motion } from "framer-motion";

export default function BotControlPanel({
  botSettings,
  onApplyBotSettings,
  onStart,
  onStop,
  onRestart,
}) {
  const [strategy, setStrategy] = useState("fvg_mid_zone");
  const [riskPct, setRiskPct] = useState(Number(botSettings?.RISK_PER_TRADE || 0.02) * 100);
  const [slPct, setSlPct] = useState(Number(botSettings?.SL_NEAREST_LEVEL_BUFFER_PCT || 0.15));
  const [tpPct, setTpPct] = useState(Number(botSettings?.TP_HVN_OFFSET_PCT || 0.12));

  return (
    <div className="glass-panel rounded-2xl p-4">
      <div className="mb-2 text-sm font-semibold text-gray-100">Bot Control Panel</div>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <label className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs text-gray-300">
          Strategy
          <select
            value={strategy}
            onChange={(event) => setStrategy(event.target.value)}
            className="mt-1 w-full bg-transparent text-sm text-gray-100 outline-none"
          >
            <option value="fvg_mid_zone">FVG Mid-zone ICT</option>
            <option value="fvg_reaction">FVG Reaction</option>
            <option value="hybrid">Hybrid</option>
          </select>
        </label>

        <label className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs text-gray-300">
          Position risk % ({riskPct.toFixed(2)}%)
          <input
            type="range"
            min="0.1"
            max="5"
            step="0.1"
            value={riskPct}
            onChange={(event) => setRiskPct(Number(event.target.value))}
            className="mt-2 w-full accent-lime-400"
          />
        </label>

        <label className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs text-gray-300">
          Stop buffer % ({slPct.toFixed(2)}%)
          <input
            type="range"
            min="0.05"
            max="0.5"
            step="0.01"
            value={slPct}
            onChange={(event) => setSlPct(Number(event.target.value))}
            className="mt-2 w-full accent-lime-400"
          />
        </label>

        <label className="rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-xs text-gray-300">
          TP offset % ({tpPct.toFixed(2)}%)
          <input
            type="range"
            min="0.05"
            max="0.5"
            step="0.01"
            value={tpPct}
            onChange={(event) => setTpPct(Number(event.target.value))}
            className="mt-2 w-full accent-lime-400"
          />
        </label>
      </div>

      <div className="mt-4 flex flex-wrap gap-2">
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.97 }}
          onClick={onStart}
          className="rounded-xl border border-emerald-300/30 bg-emerald-300/10 px-3 py-2 text-xs text-emerald-100"
        >
          Start Bot
        </motion.button>
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.97 }}
          onClick={onStop}
          className="rounded-xl border border-rose-300/30 bg-rose-300/10 px-3 py-2 text-xs text-rose-100"
        >
          Stop Bot
        </motion.button>
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.97 }}
          onClick={onRestart}
          className="rounded-xl border border-white/20 bg-white/10 px-3 py-2 text-xs text-gray-100"
        >
          Restart Bot
        </motion.button>
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.97 }}
          onClick={() =>
            onApplyBotSettings(
              {
                RISK_PER_TRADE: riskPct / 100,
                SL_NEAREST_LEVEL_BUFFER_PCT: slPct,
                TP_HVN_OFFSET_PCT: tpPct,
                STRATEGY_PROFILE: strategy,
              },
              true,
            )
          }
          className="rounded-xl border border-lime-400/40 bg-lime-400/12 px-3 py-2 text-xs text-lime-100"
        >
          Apply Control Settings
        </motion.button>
      </div>
    </div>
  );
}
