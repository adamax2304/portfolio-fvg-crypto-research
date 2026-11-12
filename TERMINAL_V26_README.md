# FVG Terminal v2.6 (FastAPI + React)

## Run Backend

```powershell
cd C:\Users\adama\OneDrive\fvgbot
py -m pip install -r requirements.txt
py -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

## Run Frontend

```powershell
cd C:\Users\adama\OneDrive\fvgbot\frontend
npm install
npm run dev
```

Open: `http://127.0.0.1:5173`

## Key WS Streams

- `ws://127.0.0.1:8000/ws/price`
- `ws://127.0.0.1:8000/ws/trades`
- `ws://127.0.0.1:8000/ws/logs`
- `ws://127.0.0.1:8000/ws/pnl`

## Important Tabs

- Dashboard
- Trading Panel
- Advanced Charts
- Analytics
- Bot Log
- Bot Settings
- Visual Settings

## Notes

- The terminal reads existing `fvg_zones.json` and `config.json`.
- Bot settings are applied through `config.json` and can trigger bot restart from UI.
- Visual settings are dashboard-only and do not change strategy logic unless you edit bot settings tab.

