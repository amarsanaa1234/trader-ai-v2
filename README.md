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
│   ├── users.py, account.py хэрэглэгчид (.env USER<n>_...), хүн бүрийн данс
│   ├── auth.py, web.py     нэвтрэлт, цэстэй вэб (бүх зах зээл нэг дор)
│   ├── notifier.py, telegram_bot.py, messages.py, dashboard.py
│   └── storage.py          зах зээл бүр өөрийн өгөгдлийн сантай
├── markets/
│   ├── base.py             📐 залгуурын стандарт (Market class)
│   └── crypto/             ₿ Binance
│       ├── config.yaml     койн, timeframe, шимтгэл, мэдээний эх сурвалж
│       ├── adapter.py      5 криптогийн эх сурвалж (funding, F&G, олон нийт, top, OI)
│       ├── data.py, broker.py, carry.py
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

## Хэрэглэгчид ба вэб

`.env` дотор `USER1_`, `USER2_`… гэж нэмнэ (`.env.example`-ийг харна уу). Бот 5 минутын дотор шинэ хүнийг автоматаар олж, Telegram-аар мэдэгдэнэ.

```bash
.venv\Scripts\python.exe main.py users                  # хэн, аль зах зээлд, түлхүүр, нууц үг
.venv\Scripts\python.exe main.py user password amar     # вэбийн нууц үг (hash хэлбэрээр хадгална)
.venv\Scripts\python.exe main.py web                    # http://127.0.0.1:8765
```

- **Нэг тархи, олон гар:** дохиог зах зээл бүрт нэг удаа бодоод, хүн бүрийн данс дээр өөрийнх нь мөнгө, эрсдэл, хязгаараар гүйцэтгэнэ. Нэг хүний алдаа бусдыг зогсоохгүй.
- **Вэб:** нэвтрэлт, цэс (Нүүр · зах зээлүүд · Тайлан · Тохиргоо · Админ), хүн бүр зөвхөн өөрийн дансаа харна. Админ хүн бүрийн дансыг харж болно. Вэб зөвхөн **үзэх** зориулалттай. Арилжааг Telegram-аар удирдана.
- **Telegram:** хүн бүр өөрийн chat ID-аар зөвхөн өөрийн дансыг удирдана. Админд нэмэлт `/users`, `/emergency` командууд бий.
- **Аюулгүй байдал:** нууц үг scrypt hash, 5 удаа буруу оруулбал 15 минут түгжинэ, нууц үг солиход хуучин session-ууд хүчингүй болно.
- **Нийтийн хуудас** `/public`: мөнгөн дүнгүй, зөвхөн хувиар.
- v1-ээс шилжүүлсэн өгөгдлийг гол админд өгөх: `python main.py crypto migrate`

## Командууд

```bash
.venv\Scripts\python.exe main.py crypto backtest   # 4 хугацаа + vault
.venv\Scripts\python.exe main.py crypto run        # ботыг ажиллуулах
.venv\Scripts\python.exe main.py crypto help       # бүх команд
```

- Ерөнхий командууд: `sync, backtest, library, research, improve, shadows, news, events, check, status, report, dashboard, telegram, resetpeak, run`
- Криптогийн нэмэлт командууд: `carry, liquidations, liqrecord`

## Шалгалтууд

```bash
.venv\Scripts\python.exe -m pytest                 # 27 тест: risk guard, оноо, backtest, залгуур, хэрэглэгч, нэвтрэлт, вэбийн эрх
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

- v1 (`crypto_ai`)-ийн бүх кодыг шинэ бүтэц рүү шилжүүлсэн. Стратеги 2 (зөвхөн крипто) болон liquidation цуглуулагч ч орсон.
- **Parity:** ижил өгөгдөл дээр v1 ба v2-ийн backtest, walk-forward, vault болон 5 койны 87,733 дохио **яг ижил** гарсан.
- 19 автомат тест. Нэг жинхэнэ алдаа олдож засагдсан: яг −10.000% уналтад эрсдэл хагасрахгүй байсан (бутархай тооны дугуйрлаас болсон).
- Сервер дээр ажиллаж буй v1 бот **хөндөгдөөгүй**.

## 1-р шатны дүн (2026-10-13)

- `.env` дээрх `USER<n>_` тохиргоогоор олон хэрэглэгч, автоматаар илрүүлнэ.
- "Нэг тархи, олон гар" trader: 2 туршилтын данс дээр ижил дохиог тус тусын эрсдэлээр гүйцэтгэсэн (0.5% эрсдэлтэй хүний position яг хагас).
- Нэвтрэлттэй, цэстэй вэб. Хүн бүр зөвхөн өөрийнхийгөө харна (вэб ба тестээр шалгасан). Админ хуудас, нийтийн хуудас, утсанд тохирсон.
- **PostgreSQL-ийг хойшлуулсан:** одоогоор зах зээл бүр өөрийн SQLite файлтай, вэб ба бот тусдаа процесс тул энэ нь 5 хэрэглэгчид хангалттай. Олон процесс нэг файл руу маш их бичиж эхлэх үед (4+ зах зээл) шилжинэ.
