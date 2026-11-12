import { useEffect } from "react";
import { useStore } from "../store/useStore";

export default function LogsPage() {
  const { logs, fetchLogs } = useStore();

  useEffect(() => {
    fetchLogs();
    const iv = setInterval(fetchLogs, 5000);
    return () => clearInterval(iv);
  }, []);

  return (
    <div className="space-y-4 animate-fade-in">
      <div className="flex items-center justify-between">
        <h1 className="text-lg font-bold text-gray-100">Bot Logs</h1>
        <button onClick={fetchLogs} className="btn-secondary text-xs">Refresh</button>
      </div>

      <div className="card p-4">
        <div className="h-[calc(100vh-200px)] overflow-y-auto scroll-y font-mono text-xs leading-relaxed">
          {logs.length > 0 ? (
            logs.map((line, i) => {
              let color = "text-gray-400";
              if (typeof line === "string") {
                if (line.includes("ERROR") || line.includes("error")) color = "text-loss";
                else if (line.includes("WARN") || line.includes("warn")) color = "text-warn";
                else if (line.includes("TRADE") || line.includes("SIGNAL")) color = "text-accent";
                else if (line.includes("ENTRY") || line.includes("EXIT")) color = "text-profit";
              }
              return (
                <div key={i} className={`py-0.5 ${color}`}>
                  {typeof line === "string" ? line : JSON.stringify(line)}
                </div>
              );
            })
          ) : (
            <div className="text-gray-600 text-center py-8">No logs available</div>
          )}
        </div>
      </div>
    </div>
  );
}
