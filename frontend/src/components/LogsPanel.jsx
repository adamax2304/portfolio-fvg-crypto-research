import { useMemo } from "react";

function classify(line) {
  const text = String(line || "").toLowerCase();
  if (text.includes("error") || text.includes("exception")) return "text-rose-300";
  if (text.includes("warn")) return "text-amber-200";
  if (text.includes("info") || text.includes("success")) return "text-emerald-300";
  return "text-gray-300";
}

export default function LogsPanel({ lines }) {
  const reversed = useMemo(() => [...(lines || [])].reverse(), [lines]);
  return (
    <div className="glass-panel h-[420px] rounded-2xl p-3">
      <div className="mb-2 text-sm font-semibold text-gray-100">Bot Log Stream</div>
      <div className="scroll-y h-[370px] rounded-xl border border-white/10 bg-black/35 p-3 font-mono text-xs leading-relaxed">
        {reversed.map((line, idx) => (
          <div key={`${idx}-${line.slice(0, 12)}`} className={classify(line)}>
            {line}
          </div>
        ))}
      </div>
    </div>
  );
}

