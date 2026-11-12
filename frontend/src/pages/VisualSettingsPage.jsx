import VisualSettingsPanel from "../components/VisualSettingsPanel";

export default function VisualSettingsPage({
  visualSettings,
  chartToggles,
  glowIntensity,
  onApplyVisual,
  onToggleChange,
  onGlowChange,
}) {
  return (
    <VisualSettingsPanel
      visualSettings={visualSettings}
      chartToggles={chartToggles}
      glowIntensity={glowIntensity}
      onApplyVisual={onApplyVisual}
      onToggleChange={onToggleChange}
      onGlowChange={onGlowChange}
    />
  );
}

