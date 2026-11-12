import { useEffect, useRef } from "react";
import { createChart, ColorType } from "lightweight-charts";

export default function MiniChart({ symbol, candles = [], signal = null }) {
  const containerRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current || !candles.length) return;

    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
    }

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: containerRef.current.clientHeight,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#8494AD",
        fontFamily: "Inter, sans-serif",
        fontSize: 10,
      },
      grid: {
        vertLines: { color: "rgba(42,54,80,0.2)" },
        horzLines: { color: "rgba(42,54,80,0.2)" },
      },
      rightPriceScale: { borderColor: "rgba(42,54,80,0.3)" },
      timeScale: { borderColor: "rgba(42,54,80,0.3)", timeVisible: true },
      crosshair: {
        vertLine: { color: "rgba(0,212,170,0.2)", width: 1, style: 2 },
        horzLine: { color: "rgba(0,212,170,0.2)", width: 1, style: 2 },
      },
    });

    const candleSeries = chart.addCandlestickSeries({
      upColor: "#0ECB81",
      downColor: "#F6465D",
      borderUpColor: "#0ECB81",
      borderDownColor: "#F6465D",
      wickUpColor: "#0ECB81",
      wickDownColor: "#F6465D",
    });

    const formatted = candles
      .map((c) => ({
        time: Math.floor(new Date(c[0] || c.time).getTime() / 1000),
        open: Number(c[1] || c.open),
        high: Number(c[2] || c.high),
        low: Number(c[3] || c.low),
        close: Number(c[4] || c.close),
      }))
      .sort((a, b) => a.time - b.time);

    if (formatted.length > 0) {
      candleSeries.setData(formatted);

      // Add signal markers
      if (signal) {
        const markers = [];
        const lastTime = formatted[formatted.length - 1].time;

        if (signal.entry_price) {
          // Entry marker
          const entryLine = chart.addLineSeries({
            color: "#00D4AA",
            lineWidth: 1,
            lineStyle: 2, // dashed
            priceLineVisible: false,
            lastValueVisible: false,
          });
          const entryData = formatted.slice(-20).map((c) => ({
            time: c.time,
            value: signal.entry_price,
          }));
          if (entryData.length) entryLine.setData(entryData);
        }

        if (signal.sl_price) {
          const slLine = chart.addLineSeries({
            color: "#F6465D",
            lineWidth: 1,
            lineStyle: 2,
            priceLineVisible: false,
            lastValueVisible: false,
          });
          const slData = formatted.slice(-20).map((c) => ({
            time: c.time,
            value: signal.sl_price,
          }));
          if (slData.length) slLine.setData(slData);
        }

        if (signal.tp_price) {
          const tpLine = chart.addLineSeries({
            color: "#0ECB81",
            lineWidth: 1,
            lineStyle: 2,
            priceLineVisible: false,
            lastValueVisible: false,
          });
          const tpData = formatted.slice(-20).map((c) => ({
            time: c.time,
            value: signal.tp_price,
          }));
          if (tpData.length) tpLine.setData(tpData);
        }

        // Arrow marker for entry direction
        if (signal.signal_side || signal.direction) {
          const dir = (signal.signal_side || signal.direction || "").toUpperCase();
          markers.push({
            time: lastTime,
            position: dir === "LONG" ? "belowBar" : "aboveBar",
            color: dir === "LONG" ? "#0ECB81" : "#F6465D",
            shape: dir === "LONG" ? "arrowUp" : "arrowDown",
            text: dir === "LONG" ? "BUY" : "SELL",
          });
        }
        if (markers.length) candleSeries.setMarkers(markers);
      }

      chart.timeScale().fitContent();
    }

    chartRef.current = chart;

    const ro = new ResizeObserver(() => {
      if (containerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        });
      }
    });
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
    };
  }, [candles, signal]);

  if (!candles.length) {
    return (
      <div className="flex items-center justify-center h-full text-gray-600 text-xs">
        No chart data
      </div>
    );
  }

  return <div ref={containerRef} className="w-full h-full" />;
}
