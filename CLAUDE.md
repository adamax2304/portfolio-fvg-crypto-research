# FVG Bot — Project Context

> **Для нового Claude на другом ПК**: этот файл — контекст MTF бота на Bybit.
> Пользователь — опытный разработчик, общение на русском, минимум лишних вопросов.
>
> **News Trader** перенесён в отдельный проект:
> `C:\Users\adama\Documents\zeph ai projects\news_trader\`

---

## Структура проекта

```
fvgbot/
├── multi_agent/          # MTF trading bot (Bybit, LIVE)
│   ├── agents/           # BBSqueezeAgent, FVGAgent, SMCAgent
│   ├── shared/           # BybitClient, StateManager, TelegramNotifier
│   ├── modules/          # TelegramCommandHandler
│   ├── config.py         # Все настройки
│   └── main.py           # Точка входа
└── requirements.txt
```

---

## ПРОЕКТ 1: V7 Portfolio Bot (multi_agent/)

### Активный агент: V7Agent (запущен 2026-05-03)
**Файл**: `multi_agent/agents/v7_agent.py`
**4 свинг стратегии (730d backtest: +1525% ret, 23.8% DD, 168 trades):**
- ST_PIVOT_4H: SuperTrend(10,3) flip → SL at pivot ± 0.3×ATR, TP=2.5R
- SMA_CROSS_1D: SMA50/200 golden/death cross → SL at pivot, TP=4.0R
- RSI_REV_4H: RSI<20 cross up / RSI>80 cross down → SL=2×ATR, TP=3R
- FIB_PUMP_4H: consolidation → pump → 38-62% retrace → SL=0.5×ATR, TP=ext 1.0

**Параметры**: risk=4%, max_pos=5, leverage≤25×, margin≤10% balance
**Фильтры**: pivot dist <7×ATR, corr >0.80 (max 1), FOMC blackout 24h, SL cooldown 4h, bear_mode=long_only

### Отключённые агенты (не удалять!)
- MTFAgent (BBS_1H_5M, BBS_1H_15M, KC_4H_15M): `enabled=False` в config.py (отключены 2026-05-03)
- Причина отключения: открывал дублированные позиции (TRUMP×3, PAXG×3 и т.д.)

### Что это
V7 свинг-трейдинг бот на **Bybit Perpetuals** (live trading).
- **Биржа**: Bybit (не MEXC, не Binance)
- **API ключи**: в `multi_agent/config.py` (BYBIT_API_KEY / BYBIT_API_SECRET)
- **Режим**: `SIMULATION_MODE = False` — реальные деньги
- **⚠️ ВНИМАНИЕ**: Bybit аккаунт на 2026-05-03 показывает ~$0.00007 USDT (нужно пополнить!)

### Запуск (Windows)
```bash
cd "C:\Users\adama\Documents\zeph ai projects\fvgbot"
py -3 multi_agent/main.py
```
Для проверки дубликатов процессов:
```bash
tasklist | findstr python
```

### Windows Autostart (как на рабочем ПК)

На рабочем ПК MTF бот запускается автоматически при логине через:

**1. Startup папка:**
```
C:\Users\<username>\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup\FVGBot_MTF.bat
```
Содержимое `FVGBot_MTF.bat`:
```bat
@echo off
start "FVGBot_MTF" /MIN cmd /c "C:\Users\<username>\OneDrive\fvgbot\start_mtf_bot.bat"
```

**2. Основной скрипт `start_mtf_bot.bat`** (в корне проекта):
- Генерирует timestamp лога
- **Защита от дублей**: перед запуском проверяет через WQL нет ли уже `multi_agent\main.py` в процессах
- Логирует в `multi_agent\logs\autostart_YYYYMMDD_HHMMSS.log`
- Команда запуска: `py -3 multi_agent\main.py`

**На новом ПК:**
```bat
REM 1. Скопируй start_mtf_bot.bat в корень проекта (уже есть)
REM 2. Создай FVGBot_MTF.bat в Startup папке:
REM    %AppData%\Roaming\Microsoft\Windows\Start Menu\Programs\Startup\
REM    Обнови путь внутри на актуальный путь к проекту
REM 3. Или запусти вручную: start_mtf_bot.bat
```

### Агенты
1. **MTFAgent** (`mtf_agent.py`) — основной активный агент. 3 стратегии в одном классе.
2. **MasterAgent** (`master_agent.py`) — оркестратор, запускает агентов, синхронизирует с Bybit.
3. **BBSqueezeAgent**, **FVGAgent**, **SMCAgent** — старые агенты, не используются в проде.

---

## Полное описание работы MTF бота

### Общий принцип

Бот торгует **Bybit USDT Perpetual фьючерсы** по принципу **Multi-Timeframe (MTF)**:
- **HTF (1h)** — определяет направление и момент входа (сигнал)
- **LTF (5m или 15m)** — точка входа (pullback или следующий бар)

Логика: если на старшем таймфрейме сжатие + импульс → ищем вход на младшем с точным SL.

### Цикл работы (каждые 5 минут)

```
1. Синхронизация позиций с Bybit (каждые 30с через MasterAgent)
2. Управление открытыми позициями:
   - Проверка SL/TP (Bybit закрыл сам? → записать в лог + TG)
   - Break-even: если прибыль ≥ 1.5R → сдвинуть SL в безубыток
3. Сканирование 87 символов на 1h сигналы
4. Для символов с сигналом → ждать вход на 5m/15m
5. Логирование воронки: total → cooldown → position → locked → pending → scanned → signal
```

### Символы (87 штук)

Отбираются динамически каждые 4 часа через 3 фильтра:
1. **Binance Futures топ-250** по объёму (обновляется каждые 4ч через Binance FAPI)
2. **Bybit ≥$500k/сутки** объём (только ликвидные)
3. **CoinGecko топ-250** по капитализации (только крупные монеты)

Итог: ~80-90 символов USDT Perpetual на Bybit.

---

### Стратегия 1: BBS_1H_5M (BB Squeeze → 5m pullback)

**Принцип:** сжатие на 1h + разжатие с импульсом → вход на откате к EMA21 на 5m

**HTF (1h) — условия сигнала:**
| Индикатор | Параметр | Условие |
|-----------|----------|---------|
| BB Squeeze | min=8 баров | BB внутри KC ≥8 баров подряд, затем выход |
| ADX(14) | ≥15 | Тренд есть (не боковик) |
| Volume | ≥1.0× MA | Объём не ниже среднего |
| Momentum | знак | Направление входа (+ = LONG, − = SHORT) |
| EMA stack | 21/50/200 | LONG: close>EMA21>EMA50>EMA200; SHORT: зеркально |
| Daily EMA200 | on | Дополнительный фильтр: дневной тренд совпадает |
| Head-fake filter | on | Если моментум разворачивается через 1 бар → сигнал отменяется |

**LTF (5m) — точка входа (pullback_ema21):**
- Ждём откат цены к EMA21 на 5m (до 20 циклов = ~100 мин)
- LONG: цена касается EMA21 снизу (low ≤ EMA21, close выше)
- SHORT: цена касается EMA21 сверху (high ≥ EMA21, close ниже)
- Вход: **market order** (лимитка не подходит — цена должна коснуться EMA21)

**SL/TP:**
- SL = ATR(14) × 1.5 от цены входа
- TP = риск × 3.0R

**Результаты (backtest v12, OOS 73d):** WR=93%, PF=10.56, 30 сделок

---

### Стратегия 2: BBS_1H_15M (BB Squeeze → 15m next bar)

**Принцип:** сжатие на 1h → вход на открытии следующего 15m бара

**HTF (1h) — условия сигнала:**
| Индикатор | Параметр | Условие |
|-----------|----------|---------|
| BB Squeeze | min=4 бара | Более ранний вход, меньшее сжатие |
| ADX(14) | ≥15 | Тренд есть |
| Volume | ≥1.0× MA | Объём норм |
| Momentum | знак | Направление |
| Head-fake filter | on | Разворот моментума → отмена |
| Daily EMA200 | off | Не используется (ухудшает PF) |

**LTF (15m) — точка входа (next_bar):**
- Вход на открытии ближайшего 15m бара после сигнала
- **Post-only limit order** по цене open этого бара (maker fee 0.020%)
- Ждём fill до 15 мин (3 цикла). Не заполнился → cancel
- Если биржа отклоняет post-only → fallback на market order

**SL/TP:**
- SL = ATR(14) × 1.5 от цены входа
- TP = риск × 2.5R

**Результаты (backtest v12, OOS 73d):** WR=61%, PF=2.55, 99 сделок

---

### Стратегия 3: KC_1H_15M (Keltner Squeeze → 15m next bar)

**Принцип:** Keltner Channel squeeze (другой тип сжатия) на 1h → вход на 15m

**HTF (1h) — условия сигнала:**
| Индикатор | Параметр | Условие |
|-----------|----------|---------|
| KC Squeeze | min=10 баров | BB внутри KC(1.5×ATR) ≥10 баров → выход |
| ADX(14) | ≥12 | Чуть мягче чем BBS |
| Volume | ≥1.0× MA | Объём норм |
| Momentum slope | 6 баров | Наклон цены за 6 баров (polyfit) = направление |

**Отличие от BBS:** Squeeze определяется иначе — BB(20) по ширине сравнивается с KC(EMA20, 1.5×ATR10). Momentum = линейная регрессия цены за 6 баров, а не TTM.

**LTF (15m):** аналогично BBS_1H_15M — post-only limit на открытии следующего бара.

**SL/TP:**
- SL = ATR(14) × 1.5
- TP = риск × 2.0R

**Результаты (backtest v12, OOS 73d):** WR=80%, PF=1.34, 15 сделок

> ⚠️ **ОТКЛЮЧЕНА** (enabled=False, 2026-04-25): OOS PF=1.35 — почти breakeven.
> Scenario тест: без KC_1H_15M → PF 2.24→2.57, OOS PF 2.20→2.49 (лучше без неё).
> Запланирована замена на KC_4H_15M (OOS PF=4.14).

---

### Стратегия 4: KC_4H_15M (KC squeeze 4h → 15m next_bar) ✅ LIVE

**Статус:** Validated в backtest + реализована в агенте (2026-04-25).
- `mtf_agent.py`: добавлен `STRAT_KC_4H_15M`, `_detect_kc_signal_4h()`, 4h fetch в `_scan_symbol_1h()`
- `config.py`: `enabled=True`

**Backtest (2026-04-25, 40 symbols, TOP_50, 730d данные):**
- IS (80%): 57 сделок, WR=73.7%, PF=4.30, DD=4.6%
- OOS (20%): 22 сделки, WR=72.7%, PF=4.14 — минимальная деградация
- Лучший конфиг: SL=1.5×ATR, TP=2.0R

**Параметры:** min_squeeze=8, adx_min=15, vol_mult=1.0, mom_bars=6, SL=1.5×ATR, TP=2.0R

**Результаты исследования TP/SL grid (backtest_new_strategies.py):**

| Стратегия | IS PF | OOS PF | IS WR | OOS WR | DD | Вердикт |
|-----------|-------|--------|-------|--------|-----|---------|
| KC_4H_15M (SL=1.5, TP=2.0) | 4.30 | **4.14** | 73.7% | 72.7% | 4.6% | ✓ ПОДТВЕРЖДЕНО |
| BBS_4H_5M (SL=2.0, TP=1.5) | 2.87 | 3.07 | 75.8% | 88.9% | 3.1% | ~ Мало сделок (9 OOS) |
| BBS_1H_15M_PB (pullback) | 1.11 | 0.99 | 51.4% | 50.8% | 25% | ✗ НЕ РАБОТАЕТ |

**Вывод:** pullback к 15m EMA21 после 1h сигнала не работает — к моменту отката импульс уже ушёл.
Текущий BBS_1H_15M (next_bar) — правильный подход.

---

### Риск-менеджмент

| Параметр | Значение | Описание |
|----------|----------|----------|
| Риск на сделку | **2%** от баланса | Фиксированный % от текущего депозита |
| Мин. маржа | **$5** | Байбит минимум — блокирует сделки при маленьком балансе |
| Макс. плечо | **50×** | Используется автоматически для соблюдения min margin |
| Макс. маржа на сделку | **5%** баланса | Ограничение концентрации |
| Макс. одновременных позиций | **4** | По всем стратегиям |
| Макс. в одну сторону | **3** | Не более 3 LONG или 3 SHORT одновременно |
| Макс. общая экспозиция | **25%** баланса | Суммарная маржа всех позиций |
| Break-even | **1.5R** | При прибыли 1.5R SL переносится в 0+0.1% |
| SL кулдаун | **4 часа** | После стопа символ не торгуется 4ч |
| Дневной стоп | **4%** убытка | Стоп торговли на день если -4% депо |

### Фильтры входа

- **Funding rate**: если ставка >+0.05%/8ч для LONG или <-0.05%/8ч для SHORT → пропуск (толпа в ту же сторону)
- **Символ заблокирован**: другой агент торгует этот символ → пропуск
- **Уже есть позиция**: в этом символе позиция открыта → пропуск
- **Лимит позиций**: стратегия или портфель на максимуме → пропуск

### Индикаторы (детально)

**TTM Squeeze (BB Squeeze):**
- BB(20, 2σ) — Bollinger Bands
- KC(EMA20, ATR14 × 1.5) — Keltner Channels
- Squeeze ON = BB внутри KC (BB_upper < KC_upper AND BB_lower > KC_lower)
- Momentum = close − LinReg(close, 20) — отклонение от линейной регрессии

**ADX(14):**
- Average Directional Index — сила тренда (0-100)
- <15 = боковик, >15 = тренд, >25 = сильный тренд

**ATR(14):**
- Average True Range — средний диапазон бара за 14 периодов
- Используется для: SL distance, BE calculation, KC ширины

**EMA (21 / 50 / 200):**
- EMA21 — краткосрочный тренд (pullback target)
- EMA50 — среднесрочный тренд
- EMA200 — долгосрочный тренд (stack filter + daily direction)

### Управление позицией

1. **Открытие**: limit (post-only) или market → Bybit SL/TP ордера сразу
2. **Break-even**: при +1.5R прибыли SL двигается в entry+0.1%
3. **Закрытие**: Bybit закрывает автоматически по SL/TP, или бот вручную при обнаружении
4. **Recovery**: при перезапуске бот сверяет историю Bybit за 48ч → восстанавливает пропущенные уведомления в TG

### Telegram управление

- **/menu** — главное меню с кнопками
- **Status** — баланс, P&L, открытые позиции, кол-во символов
- **Positions** — детали всех открытых позиций
- **Signals** — текущие ожидающие сигналы (pending)
- **Agent Control** — включить/выключить MTF агента

### Комиссии на Bybit

| Тип | Ставка |
|-----|--------|
| Taker (market) | 0.055% |
| Maker (limit) | 0.020% |

`next_bar` стратегии используют **post-only limit** → maker fee 0.020% (в 2.75× дешевле).
`pullback_ema21` использует market order (нужна точная цена касания EMA21).

### Результаты бектестов v10 (365d, 20 символов, апрель 2026)

| Стратегия | PF | PnL% | DD% | WR% | WF PF |
|-----------|-----|------|-----|-----|-------|
| BBS 1h→5m pullback | **9.66** | +216% | 1.9% | 77.4% | 9.55 |
| BBS 1h→15m next | 3.92 | +593% | 4.1% | 59.7% | 3.83 |
| KC 1h→15m next | 5.76 | +93% | 2.0% | 68.7% | 9.70 |
| Swing KC 4h→15m | 5.68 | +28% | 2.5% | 67.9% | 6.44 |

**Portfolio (4 стратегии): PnL=+189%, DD=2.1%, $1000→$2895**

**Ключевые инсайты:**
- FVG стратегия = мёртвая на крипте (все TF, все параметры)
- Multi-TF критичен: HTF сигнал (1h/4h) + LTF вход (5m/15m) = PF 4-10x
- ATR1.5 SL вместо ATR2.5 → почти удваивает PF
- Все 10 WF тестов прошли на 365d данных

**Сигнальные параметры (v10 optimized):**
- BBS 1h→5m: sq=8, adx=15, vol=1.0, pullback_ema21, wait=20
- BBS 1h→15m: sq=4, adx=15, next_bar
- KC 1h→15m: sq=10, adx=12, mom=6, next_bar
- Swing KC 4h→15m: sq=8, adx=15, mom=6, next_bar

**Текущий прод конфиг (v8, нужен апгрейд до v10):**
- BBS 15m: sq=10, adx=22, vol=1.0, ATR2.5 SL, 3.5R TP

### Критические баги (уже исправлены в коде)
1. **Дублирующий процесс** — всегда проверяй перед запуском
2. **Timestamp sync** (`retCode:10002`) — авто-ресинк каждые 2 мин в BybitClient
3. **Log rotation PermissionError** — лимит 50MB (Windows/OneDrive совместимость)

### Telegram бот
- Inline keyboard: Status | Positions | Signals | Agent Control
- Enable/disable агентов через кнопки (long-poll, 30s timeout)

---

## ПРОЕКТ 2: News Trader

> ⚠️ Перенесён в `C:\Users\adama\Documents\zeph ai projects\news_trader\`
> Для работы с ним открывай тот проект в Claude Code.

### Что это
**Отдельный проект** для торговли на новостях через **MEXC** (не Bybit).
MEXC торгует: крипта + **золото (PAXG/XAUT)** + **нефть (USOON)** — геополитика и макро события!

**API ключи MEXC:** `mx0vgl1fIE9t5yvnib` / `458eae87d3f245b1875d8f3d5ceb527d`
**Статус:** бэктест завершён (PF=9.35), live исполнение ещё не реализовано.

### Архитектура стратегии
```
Новость → Классификатор (rules.py) → Категория + AssetClass
       → Auto-discovery: скан 1m баров в окне ±N часов
         первый бар с движением ≥ X% в нужном направлении = t0
       → Вход на закрытии t0
       → SL = ATR × sl_mult (per AssetClass)
       → TP = ближайший пивот (если RR ≥ 1.5)
       → Trail: FIXED | EMA21 (MEDIUM) | Chandelier ATR×3 (STRONG)
```

### Источники данных
- **Историческая**: Binance public API (`core/binance_data.py`) — 5+ лет 1m баров
  - BTC/ETH/SOL/etc. и PAXG (золото, с 2020г.) — ок
  - Нефть: НЕТ аналога на Binance → только live через MEXC
- **Live**: MEXC ccxt wrapper (`core/mexc_client.py`)
  - MEXC имеет только ~30 дней истории → для бектестов не используем
- **Важный баг найден и исправлен**: Event dataclass имел неправильный порядок полей
  (`auto_discover` стоял ДО `discover_window_hours`), из-за чего все окна были 0.5ч вместо 24ч

### Торгуемые активы
| Класс | Символы MEXC | Триггеры |
|-------|-------------|----------|
| crypto_major | BTC/USDT, ETH/USDT... | FOMC, CPI, ETF, листинги, хаки |
| metal_gold | GOLD(PAXG)/USDT, GOLD(XAUT)/USDT | Война, FOMC, CPI |
| energy_oil | OIL(USOON)/USDT | ОПЕК, EIA, геополитика (live only!) |

### Результаты бектеста v1 (event_study.py, 17 апреля 2026)
32 события 2022–2024, данные Binance 1m:

```
ИТОГ: 25 сделок (7 скип: 4 no_trigger, 3 oil нет данных)
PF = 9.35 | WR = 44% | avg R = +3.01 | best = +33.7R | worst = -1.0R
```

### Результаты расширенного бектеста v3 (combined 2022-Apr 2026, 19 апреля 2026)
65 событий с category whitelist + tier-based SL + confirm_bars:

```
ИТОГ: 36 сделок (29 скип: 7 oil, 6 no_trigger, 5 skip_in_backtest, 4 fetch_error, 2 fomc_filtered, 2 deescalation_filtered, 2 macro_other, 1 insuf)
PF = 15.44 | WR = 75.0% | avg R = +2.24 | best = +18.74R | worst = -1.0R
```

| Категория | n | WR% | PF | avg R |
|-----------|---|-----|----|-------|
| **crypto_hack** | **12** | **83.3%** | **29.79** | **+2.4R** |
| crypto_etf | 2 | 100% | ∞ | +2.2R |
| geo_sanctions | 2 | 100% | ∞ | +1.1R |
| crypto_listing | 1 | 100% | ∞ | +0.5R |
| macro_fomc | 7 | 71.4% | 6.06 | +1.4R |
| crypto_tokenomics | 3 | 66.7% | 1.14 | +0.0R |
| geo_war | 9 | 55.6% | 18.12 | +3.8R |

| Asset Class | n | WR% | PF | avg R |
|-------------|---|-----|----|-------|
| crypto_major | 16 | **87.5%** | 18.68 | +2.2R |
| crypto_small | 11 | 72.7% | 11.54 | +1.5R |
| metal_gold | 9 | 55.6% | 15.28 | +3.2R |

| Tier | n | WR% | PF | avg R |
|------|---|-----|----|-------|
| STRONG | 26 | 76.9% | 18.59 | +2.0R |
| MEDIUM | 10 | 70.0% | 11.77 | +2.8R |

**Топ сделки:**
- Iran drone strike Apr 2024 → PAXG LONG: **+18.7R** (chandelier trail)
- OM forced liquidation Apr 2025 → OM SHORT: **+11.9R** (Sunday low-liq, 17 wallets)
- FOMC Dec-2024 gold SHORT: **+4.9R**
- BTC ETF sell-the-news Jan 2024: **+3.9R**
- Russia-Ukraine BTC SHORT: **+6.4R**
- Iran strike → BTC SHORT: **+7.2R**
- Multichain bridge hack Jul 2023 → ETH SHORT: **+7.0R**
- FOMC Dec-2023 gold LONG: **+3.5R**

**Скипнутые (надо фиксить):**
- MELANIA/AI16Z/HYPE → fetch_error (не было на Binance в момент события)
- Israel-Iran Jun 13 2025 → no_trigger (порог 0.5% для GOLD слишком высокий)
- SOL ETF Oct 28 2025 → no_trigger (телеграфирован заранее)
- TRUMP launch → insufficient_bars (листинг только на MEXC)

**Ключевые выводы v3:**
- **WR 75% достигнут** (цель 70-80%) через: tier SL×1.8/1.3, confirm_bars=1-2, category whitelist
- **crypto_hack = главная категория** (WR=83%, PF=30, n=12) — хаки дают чёткий направленный вектор
- **crypto_major WR=87.5%** — крупные хаки и surprise FOMC на BTC/ETH/BNB работают чисто
- **geo_war: WR=55% но PF=18** — редкие события дают огромные R (Iran +18.7R = R-генератор)
- **crypto_tokenomics слабее** (PF=1.14) — нужны фильтры: размер unlock, история токена

### Параметры стопов (stops.py)
```python
CRYPTO_MAJOR: sl_atr=2.0, chand_atr=3.0, max_hold=240min
METAL_GOLD:   sl_atr=1.5, chand_atr=2.5, max_hold=360min  # золото медленнее
ENERGY_OIL:   sl_atr=2.5, chand_atr=4.0, max_hold=180min  # нефть волатильна
```

### Мульти-агентная архитектура (готова, skeletons — 19 апреля 2026)

```
news_trader/
├── main.py                  ← точка входа (py -3 -m news_trader.main [--live])
├── signal_bus/              ← SignalBus (asyncio pub/sub), RawSignal/EnrichedSignal
│   ├── bus.py               topic-based pub/sub с back-pressure
│   └── types.py             RawSignal, EnrichedSignal, SignalSource, Urgency
├── agents/
│   ├── base.py              BaseAgent (lifecycle, heartbeat, stall detection)
│   ├── master.py            MasterAgent (supervise, auto-restart, exp backoff)
│   ├── twitter/agent.py     TwitterAgent (twitterapi.io REST poll, ~135 handles)
│   ├── telegram/agent.py    TelegramAgent (Telethon → lookonchain, mlmonchain, etc.)
│   ├── onchain/agent.py     OnChainAgent (Binance forceOrder WS, pluggable watchers)
│   ├── news/agent.py        NewsAgent (RSS: Coindesk/Reuters/Bloomberg/TheBlock)
│   ├── macro/agent.py       MacroAgent (calendar-driven FOMC/CPI/NFP)
│   └── executor/
│       ├── agent.py         ExecutorAgent (enriched → MEXC spot/swap orders) ✅ DONE
│       └── position.py      LivePosition (open position state + SL/TP tracking)
├── aggregator/
│   └── aggregator.py        AggregatorAgent (dedup, classify, score, tier, emit)
├── sources/
│   ├── twitter_accounts.py  135 curated handles (S/A/B tiers)
│   └── onchain_sources.py   ~40 on-chain sources (free + paid catalog)
```

**Pipeline:**
```
Source agents (Twitter/TG/OnChain/News/Macro)
    → bus.publish("raw", RawSignal)
    → AggregatorAgent: dedup(window=600s) → classify(RuleClassifier)
                     → corroborate(≥2 sources bonus) → score(0..10)
                     → tier (STRONG≥8, MEDIUM≥7, FIXED otherwise)
    → bus.publish("enriched", EnrichedSignal)
    → ExecutorAgent (TODO) → MEXC
MasterAgent supervises all, heartbeat 30s, auto-restart with exp backoff.
```

### Что ещё НЕ реализовано (следующие шаги)
1. ~~**Executor agent**~~ ✅ **DONE** — `agents/executor/agent.py` полностью реализован
2. **Score calibration** — забектестить веса в aggregator против исторических signals
3. **twitterapi.io интеграция** — TweetFetcher реализовать (сейчас заглушка)
4. **Telethon auth** — зарегистрироваться https://my.telegram.org → TG_API_ID/HASH
5. **Walk-forward тест** — разбить 2022-24 (train) / 2025-26 (test)
6. **Улучшить пороги** — MELANIA/AI16Z fetch_error (Binance listing date), порог для GOLD слишком высокий
7. **Backtesting Oil** — нужен источник 1m данных нефти (проблема не решена)
8. **LLM классификатор** — Groq pipeline для unknown/ambiguous raw signals

### Запуск live системы
```bash
cd "C:\Users\adama\Documents\zeph ai projects\news_trader"
pip install -r requirements.txt

# Dry-run (без реальных ордеров):
py -3 -m news_trader.main

# Live trading (реальные ордера MEXC):
py -3 -m news_trader.main --live --risk-usd 10 --max-concurrent 3
```

### Запуск бектеста
```bash
cd "C:\Users\adama\Documents\zeph ai projects\news_trader"

# Legacy (2022-2024, 32 events):
py -3 -m news_trader.backtests.event_study

# Combined 2022-2026 (65 events — рекомендуется):
py -3 -m news_trader.backtests.event_study_2025 --combined
```

---

## Установка зависимостей (новый ПК)

```bash
# Python 3.12+
pip install -r requirements.txt
pip install ccxt aiohttp
```

На Windows: используй `py -3` вместо `python3`.

---

## API ключи

| Сервис | Значение |
|--------|----------|
| MEXC API Key | mx0vgl1fIE9t5yvnib |
| MEXC API Secret | 458eae87d3f245b1875d8f3d5ceb527d |
| Bybit API Key | см. multi_agent/config.py |
| Telegram Bot Token | см. multi_agent/config.py |

---

## Предпочтения пользователя
- Общение на **русском языке**
- Минимум лишних вопросов — просто делай
- Permissions: можно запускать команды, редактировать файлы, читать логи
- Не создавать markdown/README файлы без запроса
- Всегда обновляй этот CLAUDE.md после больших изменений
