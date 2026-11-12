import { useEffect, useRef } from "react";
import { createChart, ColorType } from "lightweight-charts";

export default function EquityCurve({ data = [] }) {
  const containerRef = useRef(null);
  const chartRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current || !data.length) return;

    // Cleanup previous
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
        fontSize: 11,
      },
      grid: {
        vertLines: { color: "rgba(42,54,80,0.3)" },
        horzLines: { color: "rgba(42,54,80,0.3)" },
      },
      rightPriceScale: {
        borderColor: "rgba(42,54,80,0.5)",
      },
      timeScale: {
        borderColor: "rgba(42,54,80,0.5)",
        timeVisible: true,
      },
      crosshair: {
        vertLine: { color: "rgba(0,212,170,0.3)", width: 1, style: 2 },
        horzLine: { color: "rgba(0,212,170,0.3)", width: 1, style: 2 },
      },
    });

    const series = chart.addAreaSeries({
      topColor: "rgba(0,212,170,0.25)",
      bottomColor: "rgba(0,212,170,0.02)",
      lineColor: "#00D4AA",
      lineWidth: 2,
    });

    const formatted = data
      .filter((d) => d.time)
      .map((d) => ({
        time: Math.floor(new Date(d.time).getTime() / 1000),
        value: d.balance,
      }))
      .sort((a, b) => a.time - b.time);

    if (formatted.length > 0) {
      series.setData(formatted);
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
  }, [data]);

  if (!data.length) {
    return (
      <div className="flex items-center justify-center h-full text-gray-500 text-sm">
        No equity data yet
      </div>
    );
  }

  return <div ref={containerRef} className="w-full h-full" />;
}
