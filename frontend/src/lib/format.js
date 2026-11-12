export function fmtUsd(v) {
  if (v == null || v === "") return "$0.00";
  const n = Number(v);
  if (!Number.isFinite(n)) return "$0.00";
  return "$" + n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

export function fmtPct(v, digits = 1) {
  if (v == null) return "0%";
  const n = Number(v);
  if (!Number.isFinite(n)) return "0%";
  const sign = n > 0 ? "+" : "";
  return sign + n.toFixed(digits) + "%";
}

export function fmtPrice(v) {
  if (v == null || v === "") return "0";
  const n = Number(v);
  if (!Number.isFinite(n)) return "0";
  if (Math.abs(n) >= 1000) return n.toFixed(2);
  if (Math.abs(n) >= 1) return n.toFixed(4);
  return n.toFixed(6);
}

export function fmtTime(v) {
  if (!v) return "-";
  const d = new Date(v);
  if (isNaN(d.getTime())) return "-";
  return d.toLocaleString("en-GB", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function fmtShortTime(v) {
  if (!v) return "-";
  const d = new Date(v);
  return d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
}

export function pnlColor(v) {
  const n = Number(v || 0);
  if (n > 0) return "text-profit";
  if (n < 0) return "text-loss";
  return "text-gray-400";
}

export function pnlBg(v) {
  const n = Number(v || 0);
  if (n > 0) return "bg-profit/10";
  if (n < 0) return "bg-loss/10";
  return "bg-dark-600/50";
}
