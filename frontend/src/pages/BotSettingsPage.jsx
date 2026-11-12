import BotControlPanel from "../components/BotControlPanel";
import BotSettingsPanel from "../components/BotSettingsPanel";

export default function BotSettingsPage({
  botSettings,
  onApplyBotSettings,
  onStart,
  onStop,
  onRestart,
}) {
  return (
    <div className="space-y-3">
      <BotControlPanel
        botSettings={botSettings}
        onApplyBotSettings={onApplyBotSettings}
        onStart={onStart}
        onStop={onStop}
        onRestart={onRestart}
      />
      <BotSettingsPanel botSettings={botSettings} onApply={onApplyBotSettings} />
    </div>
  );
}

