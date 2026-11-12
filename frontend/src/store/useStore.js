import { create } from "zustand";
import { toast } from "sonner";
import { apiGet, apiPost, apiPut, getWsBase } from "../lib/api";

export const useStore = create((set, get) => ({
  // ── UI state ─────────────────────────
  activeTab: "dashboard",
  sidebarOpen: true,

  // ── Data state ───────────────────────
  dashboard: {
    balance: 0,
    total_pnl: 0,
    daily_pnl: 0,
    win_rate: 0,
    total_trades: 0,
    wins: 0,
    losses: 0,
    active_positions: 0,
    positions: [],
    simulation_mode: true,
    max_exposure_pct: 8,
  },
  strategies: [],
  signals: [],
  trades: [],
  equityCurve: [],
  logs: [],
  chartData: {},

  isLoading: true,
  error: null,

  // ── WebSocket refs ───────────────────
  _ws: null,

  // ── Actions ──────────────────────────
  setTab: (tab) => set({ activeTab: tab }),
  toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),

  // ── Bootstrap ────────────────────────
  bootstrap: async () => {
    set({ isLoading: true, error: null });
    try {
      const [dash, strats, signals, trades, curve] = await Promise.all([
        apiGet("/api/v2/dashboard").catch(() => get().dashboard),
        apiGet("/api/v2/strategies").catch(() => []),
        apiGet("/api/v2/signals").catch(() => []),
        apiGet("/api/v2/trades?limit=200").catch(() => []),
        apiGet("/api/v2/equity-curve").catch(() => []),
      ]);
      set({
        dashboard: dash,
        strategies: strats,
        signals: signals,
        trades: trades,
        equityCurve: curve,
        isLoading: false,
      });
      // Start WebSocket
      get().connectWs();
    } catch (e) {
      set({ error: e.message, isLoading: false });
      console.error("Bootstrap error:", e);
    }
  },

  // ── Refresh ──────────────────────────
  refresh: async () => {
    try {
      const [dash, strats, signals, trades, curve] = await Promise.all([
        apiGet("/api/v2/dashboard"),
        apiGet("/api/v2/strategies"),
        apiGet("/api/v2/signals"),
        apiGet("/api/v2/trades?limit=200"),
        apiGet("/api/v2/equity-curve"),
      ]);
      set({ dashboard: dash, strategies: strats, signals: signals, trades: trades, equityCurve: curve });
    } catch (e) {
      console.warn("Refresh failed:", e);
    }
  },

  // ── Strategy toggle ──────────────────
  toggleStrategy: async (id) => {
    try {
      const res = await apiPut(`/api/v2/strategies/${id}/toggle`);
      set((s) => ({
        strategies: s.strategies.map((st) =>
          st.id === id ? { ...st, enabled: res.enabled } : st
        ),
      }));
      toast.success(`${id}: ${res.enabled ? "enabled" : "disabled"}`);
    } catch (e) {
      toast.error("Failed to toggle strategy");
    }
  },

  // ── Bot control ──────────────────────
  botStart: async () => {
    try {
      await apiPost("/api/bot/start");
      toast.success("Bot started");
      get().refresh();
    } catch (e) { toast.error("Failed to start bot"); }
  },
  botStop: async () => {
    try {
      await apiPost("/api/bot/stop");
      toast.success("Bot stopped");
      get().refresh();
    } catch (e) { toast.error("Failed to stop bot"); }
  },

  // ── Settings ─────────────────────────
  updateSettings: async (updates) => {
    try {
      await apiPut("/api/v2/settings", { updates });
      toast.success("Settings updated");
      get().refresh();
    } catch (e) { toast.error("Failed to update settings"); }
  },

  // ── Chart data ───────────────────────
  fetchChart: async (symbol, tf = "15m") => {
    try {
      const data = await apiGet(`/api/v2/chart/${encodeURIComponent(symbol)}?timeframe=${tf}&limit=300`);
      set((s) => ({
        chartData: { ...s.chartData, [symbol]: data },
      }));
    } catch (e) {
      console.warn("Chart fetch failed:", e);
    }
  },

  // ── Logs ─────────────────────────────
  fetchLogs: async () => {
    try {
      const data = await apiGet("/api/logs?lines=200");
      set({ logs: Array.isArray(data) ? data : data.lines || [] });
    } catch (e) { console.warn("Logs fetch failed:", e); }
  },

  // ── WebSocket ────────────────────────
  connectWs: () => {
    const ws = get()._ws;
    if (ws && ws.readyState <= 1) return; // already connected or connecting

    try {
      const socket = new WebSocket(getWsBase() + "/ws/pnl");
      socket.onmessage = (evt) => {
        try {
          const msg = JSON.parse(evt.data);
          if (msg.type === "pnl" && msg.overview) {
            set((s) => ({
              dashboard: { ...s.dashboard, ...msg.overview },
            }));
          }
        } catch {}
      };
      socket.onclose = () => {
        // Reconnect after 5s
        setTimeout(() => get().connectWs(), 5000);
      };
      socket.onerror = () => socket.close();
      set({ _ws: socket });
    } catch {}
  },

  disconnectWs: () => {
    const ws = get()._ws;
    if (ws) ws.close();
    set({ _ws: null });
  },
}));
