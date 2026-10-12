# Trader AI v2

Олон зах зээлд арилжаалдаг, өөрийгөө сайжруулдаг бот. Нэг **цөм (core)** бүх зах зээлд ижил ажиллана. Зах зээл бүр нь тусдаа **залгуур (market plugin)** байна.

> ⚠️ Туршилтын төсөл, санхүүгийн зөвлөгөө биш. Бусдын мөнгөөр арилжаалахаас өмнө хуулийн зөвлөгөө аваарай (Австралид AFSL).

Төлөвлөгөө: [docs/plan_v2.html](docs/plan_v2.html) · Хуучин ботын гарын авлага: [docs/guide.html](docs/guide.html)

## Бүтэц

```
trader_ai_v2/
├── main.py                 python main.py <зах зээл> <команд>
├── core/                   ← бүх зах зээлд ИЖИЛ
│   ├── context.py          аль зах зээл идэвхтэй байгааг заана
│   ├── strategy.py         оноо: 9 ерөнхий эх сурвалж + зах зээлийн эх сурвалжууд
│   ├── risk_guard.py       🛡 хатуу хязгаар (AI өөрчилж чадахгүй)
│   ├── backtest.py, validation.py, improver.py, shadow.py, ml.py
│   ├── trader.py           үндсэн давталт
│   ├── news.py, research.py, review.py, ai.py, events.py
│   ├── notifier.py, telegram_bot.py, messages.py, dashboard.py
│   └── storage.py          зах зээл бүр өөрийн өгөгдлийн сантай
├── markets/
│   ├── base.py             📐 залгуурын стандарт (Market class)
│   └── crypto/             ₿ Binance
│       ├── config.yaml     койн, timeframe, шимтгэл, мэдээний эх сурвалж
│       ├── adapter.py      5 криптогийн эх сурвалж (funding, F&G, олон нийт, top, OI)
│       ├── data.py, broker.py, carry.py
│       ├── strategy2.py    Стратеги 2 (shadow)
│       ├── liquidations.py Liquidation цуглуулагч
│       └── cli.py          криптогийн нэмэлт командууд
├── data/<зах зээл>/        bot.db, strategy_params.json, сан, лог (git-д орохгүй)
├── tests/                  автомат тестүүд
├── tools/parity_check.py   v1 ба v2 ижил үр дүн гаргаж байгааг шалгах
└── deploy/                 серверийн тохиргоо, service файлууд
```

## Суулгах

```bash
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
.venv\Scripts\python.exe main.py crypto sync
```

## Командууд

```bash
.venv\Scripts\python.exe main.py crypto backtest   # 4 хугацаа + vault
.venv\Scripts\python.exe main.py crypto run        # ботыг ажиллуулах
.venv\Scripts\python.exe main.py crypto help       # бүх команд
```

- Ерөнхий командууд: `sync, backtest, library, research, improve, shadows, news, events, check, status, report, dashboard, telegram, resetpeak, run`
- Криптогийн нэмэлт командууд: `carry, s2, s2run, liquidations, liqrecord`

## Шалгалтууд

```bash
.venv\Scripts\python.exe -m pytest                 # 19 тест: risk guard, оноо, backtest, залгуурын стандарт
.venv\Scripts\python.exe tools\parity_check.py ..\crypto_ai <bot.db хуулбар>
```

Код өөрчлөх бүрт `pytest` ажиллуулна. Тест унасан бол серверт гаргахгүй.

## Шинэ зах зээл нэмэх

1. `markets/<нэр>/config.yaml`: хэрэглүүр, timeframe, шимтгэл, мэдээний эх сурвалж
2. `markets/<нэр>/adapter.py`: `Market`-аас удамшиж `load_frame`, `sync`, `sync_symbol`, `price`, `make_broker`, хэрэгтэй бол `sources()`, `is_open()`-ийг бичнэ. Төгсгөлд нь `MARKET = ...`
3. `markets/__init__.py` → `AVAILABLE`-д нэрийг нэмнэ
4. `pytest`: залгуурын стандартын тест шинэ зах зээл дээр автоматаар ажиллана
5. `python main.py <нэр> sync` → `backtest` → demo

Цөмийг (`core/`) өөрчлөх шаардлагагүй.

## 0-р шатны дүн (2026-10-13)

- v1 (`crypto_ai`)-ийн бүх кодыг шинэ бүтэц рүү шилжүүлсэн. Стратеги 2 болон liquidation цуглуулагч ч орсон.
- **Parity:** ижил өгөгдөл дээр v1 ба v2-ийн backtest, walk-forward, vault болон 5 койны 87,733 дохио **яг ижил** гарсан.
- 19 автомат тест. Нэг жинхэнэ алдаа олдож засагдсан: яг −10.000% уналтад эрсдэл хагасрахгүй байсан (бутархай тооны дугуйрлаас болсон).
- Сервер дээр ажиллаж буй v1 бот **хөндөгдөөгүй**.
