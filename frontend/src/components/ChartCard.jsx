import { useMemo } from "react";
import { motion } from "framer-motion";
import Plot from "react-plotly.js";
import { fmtPrice } from "../lib/format";

function computeMa(values, period) {
  const out = [];
  for (let i = 0; i < values.length; i += 1) {
    const start = Math.max(0, i - period + 1);
    const slice = values.slice(start, i + 1);
    const avg = slice.reduce((acc, v) => acc + Number(v || 0), 0) / Math.max(slice.length, 1);
    out.push(avg);
  }
  return out;
}

function computeRsi(values, period = 14) {
  const result = [];
  let gainEma = 0;
  let lossEma = 0;
  for (let i = 0; i < values.length; i += 1) {
    const close = Number(values[i] || 0);
    const prev = i > 0 ? Number(values[i - 1] || close) : close;
    const diff = close - prev;
    const gain = Math.max(diff, 0);
    const loss = Math.max(-diff, 0);
    if (i === 0) {
      gainEma = gain;
      lossEma = loss;
    } else {
      gainEma = gainEma * ((period - 1) / period) + gain / period;
      lossEma = lossEma * ((period - 1) / period) + loss / period;
    }
    const rs = lossEma > 0 ? gainEma / lossEma : 100;
    const rsi = 100 - 100 / (1 + rs);
    result.push(Number.isFinite(rsi) ? rsi : 50);
  }
  return result;
}

function zoneColor(direction, bullish, bearish) {
  return direction === "bullish" ? bullish : bearish;
}

function fmtZoneTime(value) {
  if (!value) return "n/a";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString();
}

export default function ChartCard({
  symbol,
  chartData,
  visual,
  toggles,
  glowIntensity,
}) {
  const candles = chartData?.candles || [];
  const zones = chartData?.zones || [];

  const x = useMemo(
    () => candles.map((c) => new Date(Number(c.timestamp || 0)).toISOString()),
    [candles],
  );
  const open = useMemo(() => candles.map((c) => Number(c.open || 0)), [candles]);
  const high = useMemo(() => candles.map((c) => Number(c.high || 0)), [candles]);
  const low = useMemo(() => candles.map((c) => Number(c.low || 0)), [candles]);
  const close = useMemo(() => candles.map((c) => Number(c.close || 0)), [candles]);

  const ma20 = useMemo(() => computeMa(close, 20), [close]);
  const rsi14 = useMemo(() => computeRsi(close, 14), [close]);

  const bullishColor = visual?.bullish_color || "#39ff14";
  const bearishColor = visual?.bearish_color || "#ef4444";
  const entryColor = visual?.entry_color || "#8cff66";
  const slColor = visual?.sl_color || "#fb7185";
  const tpColor = visual?.tp_color || "#39ff14";
  const cardGlowClass = glowIntensity === "high" ? "glow-high" : "glow-low";
  const xStart = x[0];
  const xEnd = x[x.length - 1];

  const traces = useMemo(() => {
    const items = [];
    if (candles.length > 0) {
      items.push({
        type: "candlestick",
        x,
        open,
        high,
        low,
        close,
        name: "Price",
        increasing: { line: { color: "#39ff14", width: 1.1 }, fillcolor: "#39ff14" },
        decreasing: { line: { color: "#ff5b75", width: 1.1 }, fillcolor: "#ff5b75" },
        yaxis: "y",
        hovertemplate:
          "Time: %{x}<br>O: %{open:.6f}<br>H: %{high:.6f}<br>L: %{low:.6f}<br>C: %{close:.6f}<extra></extra>",
      });
      items.push({
        type: "scatter",
        mode: "lines",
        x,
        y: ma20,
        name: "MA20",
        yaxis: "y",
        line: { color: "#7cff57", width: 1.1 },
        hovertemplate: "MA20: %{y:.6f}<extra></extra>",
      });
      items.push({
        type: "scatter",
        mode: "lines",
        x,
        y: rsi14,
        name: "RSI(14)",
        yaxis: "y2",
        line: { color: "#6ddf8b", width: 1.0 },
        hovertemplate: "RSI: %{y:.2f}<extra></extra>",
      });
      if (toggles.showZones && xEnd) {
        const zonePoints = zones.map((zone) => {
          const zLow = Number(zone.low || 0);
          const zHigh = Number(zone.high || 0);
          const entry = Number(zone.entry_price || zone.mid_price || (zLow + zHigh) / 2 || 0);
          const zonePosition = Number(zone.position_pct ?? 50);
          const rr = Number(zone.risk_reward || 0);
          return {
            x: xEnd,
            y: entry,
            text:
              `Gap: ${Number(zone.gap_pct || 0).toFixed(2)}%<br>` +
              `Score: ${Number(zone.score || 0).toFixed(1)}<br>` +
              `Position in zone: ${zonePosition.toFixed(1)}%<br>` +
              `RR: ${rr.toFixed(2)}<br>` +
              `Formed: ${fmtZoneTime(zone.formed_at)}<br>` +
              `Status: ${zone.status || (zone.mitigated ? "mitigated" : "active")}`,
          };
        });
        if (zonePoints.length > 0) {
          items.push({
            type: "scatter",
            mode: "markers",
            x: zonePoints.map((p) => p.x),
            y: zonePoints.map((p) => p.y),
            text: zonePoints.map((p) => p.text),
            yaxis: "y",
            marker: {
              size: 14,
              color: "rgba(255,255,255,0.001)",
              line: { width: 0 },
            },
            hovertemplate: "%{text}<extra></extra>",
            name: "FVG Info",
          });
        }
      }
    }
    return items;
  }, [candles.length, x, open, high, low, close, ma20, rsi14, toggles.showZones, xEnd, zones]);

  const shapes = useMemo(() => {
    const items = [
      {
        type: "line",
        xref: "paper",
        yref: "y2",
        x0: 0,
        x1: 1,
        y0: 70,
        y1: 70,
        line: { color: "rgba(148,163,184,0.28)", width: 1, dash: "dot" },
      },
      {
        type: "line",
        xref: "paper",
        yref: "y2",
        x0: 0,
        x1: 1,
        y0: 30,
        y1: 30,
        line: { color: "rgba(148,163,184,0.28)", width: 1, dash: "dot" },
      },
    ];
    if (!xStart || !xEnd) return items;

    if (toggles.showZones) {
      zones.forEach((zone) => {
        const zX0 = zone.formed_at || xStart;
        const zX1 = xEnd;
        items.push({
          type: "rect",
          xref: "x",
          yref: "y",
          x0: zX0,
          x1: zX1,
          y0: Number(zone.low || 0),
          y1: Number(zone.high || 0),
          line: { width: 0 },
          fillcolor: zoneColor(zone.direction, bullishColor, bearishColor),
          opacity: 0.2,
          layer: "below",
        });
      });
    }

    if (toggles.showLevels) {
      // Show E/SL/TP only for the single best zone (highest score) with RR >= 1.0
      const validZones = zones.filter(
        (z) => Number(z.risk_reward || 0) >= 1.0 &&
               Number(z.entry_price || z.mid_price || 0) > 0
      );
      const bestZone = validZones.length > 0 ? validZones[0] : null;
      if (bestZone) {
        const entry   = Number(bestZone.entry_price || bestZone.mid_price || 0);
        const sl      = Number(bestZone.sl_price || 0);
        const tp      = Number(bestZone.tp_price || 0);
        const lineX0  = bestZone.formed_at || xStart;
        if (entry > 0) {
          items.push({
            type: "line", xref: "x", yref: "y",
            x0: lineX0, x1: xEnd,
            y0: entry, y1: entry,
            line: { color: entryColor, width: 1.2, dash: "dot" },
          });
        }
        if (sl > 0) {
          items.push({
            type: "line", xref: "x", yref: "y",
            x0: lineX0, x1: xEnd,
            y0: sl, y1: sl,
            line: { color: slColor, width: 1.2 },
          });
        }
        if (tp > 0) {
          items.push({
            type: "line", xref: "x", yref: "y",
            x0: lineX0, x1: xEnd,
            y0: tp, y1: tp,
            line: { color: tpColor, width: 1.2 },
          });
        }
      }
    }
    return items;
  }, [
    zones,
    toggles.showZones,
    toggles.showLevels,
    xStart,
    xEnd,
    bullishColor,
    bearishColor,
    entryColor,
    slColor,
    tpColor,
  ]);

  const annotations = useMemo(() => {
    if (!toggles.showLabels || !xEnd) return [];
    const out = [];
    // Only label the best zone (highest score) with RR >= 1.0
    const validZones = zones.filter(
      (z) => Number(z.risk_reward || 0) >= 1.0 &&
             Number(z.entry_price || z.mid_price || 0) > 0
    );
    const bestZone = validZones.length > 0 ? validZones[0] : null;
    if (bestZone) {
      const entry = Number(bestZone.entry_price || bestZone.mid_price || 0);
      const sl    = Number(bestZone.sl_price || 0);
      const tp    = Number(bestZone.tp_price || 0);
      const rr    = Number(bestZone.risk_reward || 0);
      if (entry > 0) {
        out.push({
          x: xEnd, y: entry, xref: "x", yref: "y",
          text: `E`, showarrow: false, xanchor: "left",
          font: { size: 10, color: entryColor },
          bgcolor: "rgba(11,11,12,0.45)",
          bordercolor: "rgba(250,204,21,0.2)", borderwidth: 1,
        });
      }
      if (sl > 0) {
        out.push({
          x: xEnd, y: sl, xref: "x", yref: "y",
          text: "SL", showarrow: false, xanchor: "left",
          font: { size: 10, color: slColor },
          bgcolor: "rgba(11,11,12,0.45)",
          bordercolor: "rgba(251,113,133,0.22)", borderwidth: 1,
        });
      }
      if (tp > 0) {
        out.push({
          x: xEnd, y: tp, xref: "x", yref: "y",
          text: `TP  RR:${rr.toFixed(1)}`, showarrow: false, xanchor: "left",
          font: { size: 10, color: tpColor },
          bgcolor: "rgba(11,11,12,0.45)",
          bordercolor: "rgba(74,222,128,0.22)", borderwidth: 1,
        });
      }
    }
    return out;
  }, [zones, toggles.showLabels, xEnd, entryColor, slColor, tpColor]);

  const layout = useMemo(
    () => ({
      dragmode: "pan",
      xaxis: {
        title: "",
        type: "date",
        rangeslider: { visible: false },
        showgrid: true,
        gridcolor: "rgba(148,163,184,0.11)",
        color: "#9ca3af",
        fixedrange: false,
      },
      yaxis: {
        domain: [0.24, 1],
        side: "right",
        showgrid: true,
        gridcolor: "rgba(148,163,184,0.11)",
        color: "#9ca3af",
        fixedrange: false,
      },
      yaxis2: {
        domain: [0, 0.2],
        side: "right",
        range: [0, 100],
        showgrid: true,
        gridcolor: "rgba(148,163,184,0.1)",
        color: "#94a3b8",
        fixedrange: false,
      },
      plot_bgcolor: "rgba(4,6,5,0.98)",
      paper_bgcolor: "rgba(4,6,5,0.0)",
      font: { color: "#cbd5e1", size: 11 },
      margin: { t: 8, r: 52, b: 26, l: 22 },
      legend: {
        orientation: "h",
        y: 1.02,
        x: 0.01,
        font: { size: 10, color: "#94a3b8" },
      },
      shapes,
      annotations,
      hovermode: "x unified",
      uirevision: `${symbol}-${chartData?.timeframe || "15m"}`,
    }),
    [shapes, annotations, symbol, chartData?.timeframe],
  );

  return (
    <motion.div
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.26 }}
      whileHover={{ y: -1, scale: 1.002 }}
      className={`glass-panel rounded-3xl p-3 ${cardGlowClass}`}
    >
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="rounded-lg bg-lime-500/25 px-2 py-1 text-xs font-semibold">
            {chartData?.short_symbol || symbol.replace("/USDT:USDT", "USDT")}
          </div>
          <div className="rounded-lg bg-white/10 px-2 py-1 text-xs text-gray-200">
            {chartData?.timeframe || "15m"}
          </div>
          <div className="rounded-lg bg-white/10 px-2 py-1 text-xs text-gray-200">
            {(zones || []).length} zones
          </div>
        </div>
        <div className="text-xs text-gray-400">
          Last: <span className="text-gray-100">{fmtPrice(chartData?.last_price)}</span>
        </div>
      </div>

      <div className="h-[410px] w-full">
        {candles.length > 0 ? (
          <Plot
            data={traces}
            layout={layout}
            config={{
              responsive: true,
              displayModeBar: true,
              modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
              scrollZoom: true,
              displaylogo: false,
            }}
            style={{ width: "100%", height: "100%" }}
            useResizeHandler
          />
        ) : (
          <div className="flex h-full items-center justify-center rounded-xl border border-white/10 bg-black/20 text-sm text-gray-400">
            No candle data yet for this symbol/timeframe.
          </div>
        )}
      </div>
    </motion.div>
  );
}
