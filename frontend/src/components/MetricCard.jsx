export default function MetricCard({ label, value, sub, color, icon: Icon }) {
  return (
    <div className="card p-4 card-glow animate-fade-in">
      <div className="flex items-start justify-between mb-2">
        <span className="text-xs font-medium text-gray-400 uppercase tracking-wide">{label}</span>
        {Icon && (
          <div className="w-8 h-8 rounded-lg bg-dark-600/50 flex items-center justify-center">
            <Icon size={16} className="text-gray-400" />
          </div>
        )}
      </div>
      <div className={`text-2xl font-bold ${color || "text-gray-100"}`}>{value}</div>
      {sub && <div className="text-xs text-gray-500 mt-1">{sub}</div>}
    </div>
  );
}
