import { create } from "zustand";
import { toast } from "sonner";
import { apiGet, apiPost, apiPut, getWsBase } from "../lib/api";

const DEFAULT_TOGGLES = {
  showZones: true,
  showLevels: true,
  showLabels: true,
  showMitigated: false,
};

const SYMBOL_AMBIENT_SOUNDS = [
  "https://assets.mixkit.co/active_storage/sfx/2574/2574-preview.mp3",
  "https://assets.mixkit.co/active_storage/sfx/2575/2575-preview.mp3",
  "https://assets.mixkit.co/active_storage/sfx/1109/1109-preview.mp3",
];

function fallbackTone() {
  try {
    const context = new window.AudioContext();
    const master = context.createGain();
    master.gain.value = 0.0001;
    master.connect(context.destination);

    const o1 = context.createOscillator();
    const o2 = context.createOscillator();
    const o3 = context.createOscillator();
    o1.type = "sine";
    o2.type = "triangle";
    o3.type = "sine";
    o1.frequency.value = 392;
    o2.frequency.value = 523.25;
    o3.frequency.value = 659.25;
    o1.connect(master);
    o2.connect(master);
    o3.connect(master);

    const now = context.currentTime;
    master.gain.exponentialRampToValueAtTime(0.028, now + 0.04);
    master.gain.exponentialRampToValueAtTime(0.0001, now + 0.42);
    o1.start();
    o2.start();
    o3.start();
    setTimeout(() => {
      o1.stop();
      o2.stop();
      o3.stop();
      context.close();
    }, 450);
  } catch {
    // No-op in restricted browsers.
  }
}

function playAmbientSymbolSound() {
  const randomUrl =
    SYMBOL_AMBIENT_SOUNDS[Math.floor(Math.random() * SYMBOL_AMBIENT_SOUNDS.length)];
  try {
    const audio = new Audio(randomUrl);
    audio.volume = 0.35;
    audio.currentTime = 0;
    audio.play().catch(() => fallbackTone());
  } catch {
    fallbackTone();
  }
}

function barsPerDay(timeframe) {
  if (timeframe === "5m") return 288;
  if (timeframe === "15m") return 96;
  if (timeframe === "1h") return 24;
  return 96;
}

function toQuery(params) {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value === undefined || value === null || value === "") return;
    query.set(key, String(value));
  });
  const txt = query.toString();
  return txt ? `?${txt}` : "";
}

export const useTerminalStore = create((set, get) => ({
  isBootstrapping: false,
  error: "",

  activeTab: "dashboard",
  rightTab: "pending",
  sidebarCollapsed: false,
  focusMode: false,
  glowIntensity: "high",
  soundEnabled: true,

  timeframe: "15m",
  historyDays: 5,
  multiMode: "grid",
  chartToggles: DEFAULT_TOGGLES,
  filters: {
    query: "",
    minScore: 0,
    minGap: 0,
  },

  overview: {},
  analytics: {},
  symbols: [],
  selectedSymbols: [],
  chartDataBySymbol: {},
  symbolInsight: null,
  pendingSignals: [],
  activePositions: [],
  tradeHistory: [],
  logs: [],
  settings: {
    bot: {},
    backtest: {},
    visual: {},
  },
  botStatus: {},

  ws: {
    price: null,
    trades: null,
    logs: null,
    pnl: null,
  },

  setActiveTab: (activeTab) => set({ activeTab }),
  setRightTab: (rightTab) => set({ rightTab }),
  setTimeframe: (timeframe) => set({ timeframe }),
  setHistoryDays: (historyDays) => set({ historyDays }),
  setMultiMode: (multiMode) => set({ multiMode }),
  setFocusMode: (focusMode) => set({ focusMode }),
  setGlowIntensity: (glowIntensity) => set({ glowIntensity }),
  setSoundEnabled: (soundEnabled) => set({ soundEnabled }),
  toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
  updateFilter: (key, value) =>
    set((state) => ({ filters: { ...state.filters, [key]: value } })),
  updateChartToggle: (key, value) =>
    set((state) => ({ chartToggles: { ...state.chartToggles, [key]: value } })),

  selectSymbol: (symbol) =>
    set((state) => {
      const selected = [...state.selectedSymbols];
      const idx = selected.indexOf(symbol);
      const symbolRow = state.symbols.find((item) => item.symbol === symbol);
      if (idx >= 0) {
        selected.splice(idx, 1);
      } else {
        selected.push(symbol);
        if (state.soundEnabled) {
          playAmbientSymbolSound();
        }
        toast.success(`${symbolRow?.short_symbol || symbol} selected - ${symbolRow?.zones_count || 0} active zones`);
      }
      return {
        selectedSymbols: selected,
        activeTab: "charts",
        symbolInsight: symbolRow
          ? {
              symbol: symbolRow.short_symbol,
              timeframe: symbolRow.timeframe,
              zones: symbolRow.zones_count,
              score: symbolRow.top_score,
              gap: symbolRow.top_gap_pct,
              price: symbolRow.last_price,
            }
          : null,
      };
    }),

  focusSymbolFromSignal: (signal) =>
    set((state) => {
      const symbol = signal?.symbol;
      if (!symbol) return {};
      const selected = [...state.selectedSymbols];
      if (!selected.includes(symbol)) selected.push(symbol);
      if (state.soundEnabled) {
        playAmbientSymbolSound();
      }
      toast.message(
        `${signal.short_symbol || symbol.replace("/USDT:USDT", "USDT")} ${signal.signal_side || signal.direction || ""}`,
      );
      return {
        selectedSymbols: selected,
        activeTab: "charts",
        rightTab: "pending",
        symbolInsight: {
          symbol: signal.short_symbol || symbol.replace("/USDT:USDT", "USDT"),
          timeframe: "15m",
          zones: 1,
          score: Number(signal.score || 0),
          gap: Number(signal.gap_pct || 0),
          price: Number(signal.entry_price || 0),
          status: signal.entry_status || signal.status || "pending",
        },
      };
    }),

  clearAllCharts: async () => {
    set({ selectedSymbols: [], chartDataBySymbol: {} });
    try {
      await apiPost("/api/charts/clear");
    } catch {
      // no-op
    }
  },

  bootstrap: async () => {
    if (get().isBootstrapping) return;
    set({ isBootstrapping: true, error: "" });
    try {
      const [
        overview,
        analytics,
        symbolsResp,
        pendingResp,
        positionsResp,
        historyResp,
        logsResp,
        settingsResp,
        botStatus,
      ] = await Promise.all([
        apiGet("/api/overview"),
        apiGet("/api/analytics"),
        apiGet("/api/market/symbols"),
        apiGet("/api/pending-signals?limit=200"),
        apiGet("/api/active-positions"),
        apiGet("/api/trade-history?limit=500"),
        apiGet("/api/logs?lines=220"),
        apiGet("/api/settings"),
        apiGet("/api/bot/status"),
      ]);

      const visual = settingsResp?.dashboard?.visual || {};
      const symbols = symbolsResp?.items || [];
      const autoSelected = symbols
        .filter((item) => item.zones_count > 0)
        .slice(0, 3)
        .map((item) => item.symbol);
      const fallbackSelected = autoSelected.length > 0 ? autoSelected : symbols.slice(0, 1).map((item) => item.symbol);
      const firstSelectedRow = symbols.find((item) => item.symbol === fallbackSelected[0]) || null;

      set({
        overview,
        analytics,
        symbols,
        pendingSignals: pendingResp?.items || [],
        activePositions: positionsResp?.items || [],
        tradeHistory: historyResp?.items || [],
        logs: logsResp?.lines || [],
        settings: {
          bot: settingsResp?.bot || {},
          backtest: settingsResp?.backtest || {},
          visual,
        },
        chartToggles: {
          showZones: visual?.show_zones ?? true,
          showLevels: visual?.show_levels ?? true,
          showLabels: visual?.show_labels ?? true,
          showMitigated: visual?.show_mitigated ?? false,
        },
        filters: {
          query: "",
          minScore: Number(visual?.threshold ?? 0),
          minGap: Number(visual?.min_gap_percent ?? 0),
        },
        selectedSymbols: fallbackSelected,
        symbolInsight: firstSelectedRow
          ? {
              symbol: firstSelectedRow.short_symbol,
              timeframe: firstSelectedRow.timeframe,
              zones: firstSelectedRow.zones_count,
              score: firstSelectedRow.top_score,
              gap: firstSelectedRow.top_gap_pct,
              price: firstSelectedRow.last_price,
            }
          : null,
        glowIntensity: visual?.glow_intensity || "high",
        botStatus,
      });

      await get().fetchChartDataForSelected();
      get().connectStreams();
    } catch (error) {
      set({ error: String(error) });
      toast.error("Failed to bootstrap dashboard");
    } finally {
      set({ isBootstrapping: false });
    }
  },

  fetchSymbols: async () => {
    const { filters, chartToggles } = get();
    const query = toQuery({
      q: filters.query,
      min_score: filters.minScore,
      min_gap: filters.minGap,
      include_mitigated: chartToggles.showMitigated,
    });
    const response = await apiGet(`/api/market/symbols${query}`);
    set({ symbols: response.items || [] });
  },

  fetchChartDataForSelected: async () => {
      const {
      selectedSymbols,
      timeframe,
      historyDays,
      filters,
      chartToggles,
      chartDataBySymbol,
    } = get();
    if (!selectedSymbols.length) return;

    const limit =
      historyDays === "max"
        ? 10000
        : Math.min(10000, Math.max(260, barsPerDay(timeframe) * Number(historyDays || 5) + 120));
    const updates = {};
    const batchSize = 6;
    for (let start = 0; start < selectedSymbols.length; start += batchSize) {
      const batch = selectedSymbols.slice(start, start + batchSize);
      // eslint-disable-next-line no-await-in-loop
      await Promise.all(
        batch.map(async (symbol) => {
          try {
            const query = toQuery({
              symbol,
              timeframe,
              limit,
              min_score: filters.minScore,
              min_gap: filters.minGap,
              include_mitigated: chartToggles.showMitigated,
            });
            const payload = await apiGet(`/api/market/chart-data${query}`);
            updates[symbol] = payload;
          } catch {
            updates[symbol] = chartDataBySymbol[symbol] || {
              symbol,
              candles: [],
              zones: [],
            };
          }
        }),
      );
    }
    set((state) => ({ chartDataBySymbol: { ...state.chartDataBySymbol, ...updates } }));
  },

  refreshAll: async () => {
    try {
      const [overview, analytics, pendingResp, positionsResp, historyResp, botStatus] =
        await Promise.all([
          apiGet("/api/overview"),
          apiGet("/api/analytics"),
          apiGet("/api/pending-signals?limit=200"),
          apiGet("/api/active-positions"),
          apiGet("/api/trade-history?limit=500"),
          apiGet("/api/bot/status"),
        ]);

      set({
        overview,
        analytics,
        pendingSignals: pendingResp?.items || [],
        activePositions: positionsResp?.items || [],
        tradeHistory: historyResp?.items || [],
        botStatus,
      });
      await get().fetchSymbols();
      await get().fetchChartDataForSelected();
      toast.success("Dashboard refreshed");
    } catch (error) {
      toast.error(`Refresh failed: ${String(error)}`);
    }
  },

  applyBotSettings: async (updates, restart = false) => {
    const payload = await apiPut("/api/settings/bot", {
      updates,
      apply_restart_scan: restart,
    });
    set((state) => ({
      settings: {
        ...state.settings,
        bot: payload?.bot || state.settings.bot,
      },
    }));
    toast.success(restart ? "Bot settings applied and bot restarted" : "Bot settings applied");
  },

  applyVisualSettings: async (updates) => {
    const payload = await apiPut("/api/settings/visual", {
      updates,
      apply_restart_scan: false,
    });
    set((state) => {
      const visual = { ...state.settings.visual, ...(payload?.visual || updates) };
      return {
        settings: {
          ...state.settings,
          visual,
        },
        chartToggles: {
          showZones: visual.show_zones ?? state.chartToggles.showZones,
          showLevels: visual.show_levels ?? state.chartToggles.showLevels,
          showLabels: visual.show_labels ?? state.chartToggles.showLabels,
          showMitigated: visual.show_mitigated ?? state.chartToggles.showMitigated,
        },
      };
    });
    toast.success("Visual settings applied");
  },

  botStart: async () => {
    const response = await apiPost("/api/bot/start");
    toast.success(response?.message || "Bot start command sent");
    set({ botStatus: response?.status || {} });
  },

  botStop: async () => {
    const response = await apiPost("/api/bot/stop");
    toast.success(response?.message || "Bot stop command sent");
    set({ botStatus: response?.status || {} });
  },

  botRestart: async () => {
    const response = await apiPost("/api/bot/restart");
    toast.success(response?.message || "Bot restarted");
    set({ botStatus: response?.status || {} });
  },

  exportCsv: () => {
    const { filters, chartToggles } = get();
    const query = toQuery({
      min_score: filters.minScore,
      min_gap: filters.minGap,
      include_mitigated: chartToggles.showMitigated,
    });
    window.open(`${import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000"}/api/export/zones.csv${query}`, "_blank");
  },

  exportPine: () => {
    const { filters, chartToggles } = get();
    const query = toQuery({
      min_score: filters.minScore,
      min_gap: filters.minGap,
      include_mitigated: chartToggles.showMitigated,
    });
    window.open(`${import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000"}/api/export/zones.pine${query}`, "_blank");
  },

  disconnectStreams: () => {
    const ws = get().ws;
    Object.values(ws).forEach((socket) => {
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.close();
      }
    });
    set({
      ws: {
        price: null,
        trades: null,
        logs: null,
        pnl: null,
      },
    });
  },

  connectStreams: () => {
    get().disconnectStreams();
    const wsBase = getWsBase();

    const priceWs = new WebSocket(`${wsBase}/ws/price`);
    const tradesWs = new WebSocket(`${wsBase}/ws/trades`);
    const logsWs = new WebSocket(`${wsBase}/ws/logs`);
    const pnlWs = new WebSocket(`${wsBase}/ws/pnl`);

    priceWs.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      if (!payload?.symbols) return;
      set((state) => {
        const priceMap = new Map(payload.symbols.map((s) => [s.symbol, s]));
        const symbols = state.symbols.map((item) => {
          const fromWs = priceMap.get(item.symbol);
          return fromWs
            ? {
                ...item,
                last_price: fromWs.last_price,
                zones_count: fromWs.zones_count,
                htf_bias: fromWs.htf_bias,
              }
            : item;
        });
        return { symbols };
      });
    };

    tradesWs.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      const oldCount = get().pendingSignals.length;
      const newCount = payload?.pending_signals?.length || 0;
      set({
        pendingSignals: payload?.pending_signals || [],
        activePositions: payload?.positions || [],
        botStatus: payload?.bot_runtime || get().botStatus,
      });
      if (newCount > oldCount) {
        toast.info(`New potential setup: +${newCount - oldCount}`);
        if (get().soundEnabled) playAmbientSymbolSound();
      }
    };

    logsWs.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      if (!Array.isArray(payload?.lines)) return;
      set((state) => ({
        logs: [...state.logs, ...payload.lines].slice(-2000),
      }));
    };

    pnlWs.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      if (payload?.overview) {
        set({ overview: payload.overview });
      }
      if (payload?.analytics) {
        set((state) => ({
          analytics: {
            ...state.analytics,
            ...payload.analytics,
          },
        }));
      }
    };

    set({
      ws: {
        price: priceWs,
        trades: tradesWs,
        logs: logsWs,
        pnl: pnlWs,
      },
    });
  },
}));
