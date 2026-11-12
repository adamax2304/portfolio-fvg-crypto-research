import { motion } from "framer-motion";

export default function TogglePill({ label, checked, onChange }) {
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      className={`relative inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs transition ${
        checked
          ? "border-lime-400/60 bg-lime-400/14 text-lime-100"
          : "border-white/10 bg-white/5 text-gray-300"
      }`}
    >
      <motion.span
        className={`h-2.5 w-2.5 rounded-full ${checked ? "bg-lime-400" : "bg-gray-500"}`}
        transition={{ duration: 0.14, ease: "easeOut" }}
        animate={{ scale: checked ? 1.02 : 1 }}
      />
      {label}
    </button>
  );
}
