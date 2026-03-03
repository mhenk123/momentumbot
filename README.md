# Hyperliquid Momentum Paper-Trading Bot

Ein selbstlernender Trading-Bot, der Live-Marktdaten von Hyperliquid nutzt und Trades rein lokal mit einem virtuellen Portfolio simuliert. Keine echten Orders, keine Wallet-Verbindung – reines Paper-Trading mit KI-gestützter Selbstoptimierung.

---

## Inhaltsverzeichnis

1. [Schnellstart](#schnellstart)
2. [Projektstruktur](#projektstruktur)
3. [Funktionsweise im Überblick](#funktionsweise-im-überblick)
4. [Trading-Logik im Detail](#trading-logik-im-detail)
5. [Risikomanagement](#risikomanagement)
6. [Machine Learning & Selbstoptimierung](#machine-learning--selbstoptimierung)
7. [Telegram-Benachrichtigungen](#telegram-benachrichtigungen)
8. [Konfigurationsreferenz](#konfigurationsreferenz)
9. [Datenbankschema](#datenbankschema)
10. [VPS-Deployment](#vps-deployment)

---

## Schnellstart

**1. Abhängigkeiten installieren**

```bash
pip install -r requirements.txt
```

**2. Telegram-Bot einrichten**

- Erstelle einen Bot über [@BotFather](https://t.me/BotFather) und kopiere den Token.
- Schreibe deinem Bot eine Nachricht, dann rufe `https://api.telegram.org/bot<TOKEN>/getUpdates` auf, um deine Chat-ID zu ermitteln.

**3. `.env` befüllen**

```env
TELEGRAM_BOT_TOKEN=1234567890:ABCdef...
TELEGRAM_CHAT_ID=987654321
```

**4. Bot starten**

```bash
python bot.py
```

Beim ersten Start wird automatisch die SQLite-Datenbank `database.db` angelegt und ein Startstatus per Telegram gesendet.

---

## Projektstruktur

```
momentumbot/
├── bot.py           # Hauptskript – gesamte Logik
├── config.json      # Parameter & virtueller Kontostand (wird vom Bot überschrieben)
├── .env             # Telegram-Credentials (nicht einchecken!)
├── requirements.txt # Python-Abhängigkeiten
├── database.db      # SQLite-Datenbank (wird automatisch erstellt)
└── bot.log          # Laufendes Log-File (wird automatisch erstellt)
```

---

## Funktionsweise im Überblick

```
┌─────────────────────────────────────────────────┐
│                  Scheduler (30s Loop)            │
│                                                  │
│  Alle 15 min  ──► run_scan_cycle()               │
│                    ├── monitor_positions()        │
│                    └── scan_market() → open_trade│
│                                                  │
│  Täglich 18:30 ──► send_daily_report()           │
│                                                  │
│  Sonntag 20:00 ──► run_ml_optimization()         │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│          Hyperliquid REST API (nur Lesezugriff)  │
│  /info  type=meta          → alle Perp-Symbole  │
│  /info  type=allMids       → aktuelle Kurse     │
│  /info  type=candleSnapshot→ OHLCV-Kerzen       │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│          Lokale Simulation (Paper-Trading)       │
│  virtual_balance_usd in config.json             │
│  Trades in database.db (SQLite)                 │
└─────────────────────────────────────────────────┘
```

---

## Trading-Logik im Detail

### 1. Markt-Scanner (`scan_market`)

Wird alle **15 Minuten** ausgeführt. Der Scanner läuft über alle aktiv gehandelten Hyperliquid-Perpetuals und berechnet für jeden Coin:

| Metrik | Berechnung |
|---|---|
| **RSI (14)** | Wilder-Methode auf die letzten 4h Schlusskurse (15m-Kerzen) |
| **Volumenanstieg** | Letzte Kerze vs. Durchschnitt aller vorherigen Kerzen im Fenster (%) |
| **Kursanstieg (4h)** | Erste vs. letzte Schlusskurs der letzten 4 Stunden (%) |
| **BTC-Trend** | 4h-Kursänderung von BTC als Markt-Referenz |

**Filter:** Coins mit RSI > `rsi_overbought_limit` (Standard: 70) werden ausgeschlossen, um überhitzte Märkte zu meiden.

**Scoring:** Aus den verbleibenden Kandidaten wird der Coin mit dem höchsten **Momentum-Score** gewählt:

```
Score = Volumenanstieg × 0.6 + Kursanstieg × 0.4
```

Bedingung: Beide Werte (Volumen und Kurs) müssen positiv sein – kein Einstieg bei gemischten Signalen.

### 2. Einstieg (`open_trade`)

Wenn ein Kandidat gefunden wird und noch ein Trade-Slot frei ist (max. 3 gleichzeitig):

- **Positionsgröße:** 10% der aktuellen `virtual_balance_usd`
- Die Größe wird sofort vom virtuellen Kontostand abgezogen
- Alle Einstiegs-Features (RSI, Volumen, BTC-Trend, Kursanstieg) werden in der SQLite-DB gespeichert – das ist die Grundlage für das spätere ML-Training
- Telegram-Benachrichtigung wird gesendet

### 3. Positions-Monitoring (`monitor_positions`)

Läuft am **Anfang jedes Scan-Zyklus** (also auch alle 15 Minuten). Prüft jede offene Position gegen den aktuellen Marktpreis:

```
Für jede offene Position:
  1. Aktuellen Preis abrufen (allMids)
  2. Höchstkurs (highest_price) aktualisieren
  3. Trailing Stop aktivieren? → wenn Gewinn ≥ trailing_stop_activation_pct
  4. Trailing Stop Niveau anheben? → wenn Kurs neues Hoch erreicht
  5. Ausstieg auslösen? → wenn Kurs ≤ Stop-Preis
```

---

## Risikomanagement

Der Bot verwendet ein **Zwei-Stufen-Stop-System**:

### Stufe 1 – Harter Stop-Loss

Wird direkt beim Einstieg gesetzt. Standardmäßig **−5%** vom Einstiegspreis.

```
Stop-Preis = Einstiegspreis × (1 - 0.05)
```

Wird der Preis erreicht, schließt der Bot die Position sofort.

### Stufe 2 – Dynamischer Trailing Stop

Aktiviert sich, sobald der Trade **+5%** im Plus liegt (`trailing_stop_activation_pct`).

Ab diesem Punkt wird der Trailing Stop verfolgt:

```
Trailing Stop = Höchstkurs × (1 - trailing_stop_distance_pct)
               = Höchstkurs × (1 - 0.03)
```

Der Trailing Stop **steigt mit** wenn der Kurs steigt, fällt aber nie ab. Sobald der Kurs den Trailing Stop unterschreitet, wird die Position geschlossen und der Gewinn gesichert.

**Beispiel:**

```
Einstieg:    $100
Hard-SL:     $95 (−5%)
Kurs steigt auf $107 → Trailing Stop aktiviert bei $103.79 (3% unter $107)
Kurs steigt auf $115 → Trailing Stop zieht nach auf $111.55
Kurs fällt auf $111  → Trailing Stop ausgelöst → Gewinn: ~+11%
```

---

## Machine Learning & Selbstoptimierung

### Wann läuft die Optimierung?

Jeden **Sonntag um 20:00 Uhr** (Europe/Berlin), sofern mindestens **10 abgeschlossene Trades** in der Datenbank vorhanden sind.

### Wie funktioniert es?

**Schritt 1 – Daten laden**

Alle geschlossenen Trades werden aus der SQLite-DB geladen. Jeder Trade enthält die Einstiegs-Features und das Ergebnis (PnL in %).

**Schritt 2 – RandomForest trainieren**

```python
Features (X):  rsi_at_entry, volume_chg_pct, btc_trend_pct, price_chg_4h
Zielgröße (y): pnl_pct  (Gewinn/Verlust in %)
```

Ein `RandomForestRegressor` (100 Bäume) lernt, welche Kombination von Einstiegs-Features zu positiven oder negativen Ergebnissen geführt hat.

**Schritt 3 – Parameter anpassen**

Die Anpassung erfolgt in kleinen Schritten (max. ±1 Punkt) und innerhalb definierter Grenzen:

| Parameter | Logik | Min | Max |
|---|---|---|---|
| `rsi_overbought_limit` | Wenn hohe RSI-Einstiege schlechter performen → senken | 55 | 80 |
| `initial_stop_loss_pct` | Wenn viele Trades den SL auslösen → enger setzen | 2% | 10% |
| `trailing_stop_distance_pct` | Wenn Gewinner kaum über Aktivierungslevel liegen → enger | 1.5% | 6% |

**Schritt 4 – config.json aktualisieren**

Die neuen Werte werden in `config.json` gespeichert. Ab dem nächsten Scan-Zyklus arbeitet der Bot mit den optimierten Parametern.

### Feature-Wichtigkeit

Nach jeder Optimierung wird im Telegram-Report angezeigt, welche Features der RandomForest als am wichtigsten eingestuft hat – so ist nachvollziehbar, worauf der Bot gerade hauptsächlich reagiert.

---

## Telegram-Benachrichtigungen

| Ereignis | Zeitpunkt | Inhalt |
|---|---|---|
| **Start** | Beim Hochfahren | Kontostand, Scan-Intervall, Report-Zeiten |
| **Trade geöffnet** | Sofort | Symbol, Preis, RSI, Volumen, Kursanstieg, Positionsgröße, Stop-Loss |
| **Trade geschlossen** | Sofort | Symbol, Einstieg/Ausstieg, PnL in USD & %, neuer Kontostand, Schließungsgrund |
| **Daily Report** | Täglich 18:30 Uhr | Kontostand, Tages-PnL, offene Positionen mit unrealisiertem PnL, Win-Rate |
| **ML-Report** | Sonntags nach Optimierung | Analysierte Trades, alte → neue Parameter, Feature-Wichtigkeiten |

---

## Konfigurationsreferenz

Alle Werte in `config.json` können manuell angepasst werden. Parameter mit `_min`/`_max`-Suffix definieren die Grenzen für die ML-Anpassung.

```jsonc
{
  // Virtuelles Konto
  "virtual_balance_usd": 100000.0,      // Aktueller Kontostand (wird vom Bot aktualisiert)
  "position_size_pct": 0.10,            // 10% des Kontostands pro Trade

  // Trade-Limits
  "max_open_trades": 3,                 // Maximale gleichzeitige Positionen

  // Scanner
  "scan_interval_minutes": 15,          // Wie oft gescannt wird
  "lookback_hours": 4,                  // Zeitfenster für RSI und Momentum
  "rsi_period": 14,                     // RSI-Periode (Kerzen)
  "rsi_overbought_limit": 70,           // Coins über diesem RSI werden ignoriert

  // Risikomanagement
  "initial_stop_loss_pct": 5.0,         // Harter Stop-Loss in % unter Einstieg
  "trailing_stop_activation_pct": 5.0,  // Gewinn, ab dem Trailing Stop aktiv wird (%)
  "trailing_stop_distance_pct": 3.0,    // Abstand des Trailing Stops vom Höchstkurs (%)

  // ML-Grenzen (Schutz vor zu aggressiven Anpassungen)
  "ml_min_trades_required": 10,         // Mindest-Trades für ML-Optimierung
  "ml_adjustment_step": 1.0,            // Maximale Anpassung pro Optimierungslauf
  "rsi_limit_min": 55,                  // RSI-Limit darf nicht unter diesen Wert
  "rsi_limit_max": 80,                  // RSI-Limit darf nicht über diesen Wert
  "stop_loss_min": 2.0,                 // Stop-Loss mindestens 2%
  "stop_loss_max": 10.0,                // Stop-Loss maximal 10%
  "trailing_stop_min": 1.5,             // Trailing Stop mindestens 1.5%
  "trailing_stop_max": 6.0,             // Trailing Stop maximal 6%

  // Zeitplanung
  "daily_report_time": "18:30",         // Uhrzeit des täglichen Reports (HH:MM)
  "ml_optimization_time": "20:00",      // Uhrzeit der sonntäglichen Optimierung
  "timezone": "Europe/Berlin"           // Zeitzone für alle geplanten Aufgaben
}
```

---

## Datenbankschema

Die SQLite-Datenbank `database.db` enthält eine Tabelle `trades`:

| Spalte | Typ | Beschreibung |
|---|---|---|
| `id` | INTEGER | Primärschlüssel |
| `symbol` | TEXT | z.B. `ETH`, `SOL` |
| `entry_price` | REAL | Kurs beim Einstieg |
| `exit_price` | REAL | Kurs beim Ausstieg |
| `size_usd` | REAL | Positionsgröße in USD |
| `entry_time` | TEXT | Zeitstempel Einstieg |
| `exit_time` | TEXT | Zeitstempel Ausstieg |
| `pnl_usd` | REAL | Gewinn/Verlust in USD |
| `pnl_pct` | REAL | Gewinn/Verlust in % |
| `status` | TEXT | `open` oder `closed` |
| `rsi_at_entry` | REAL | RSI beim Einstieg (ML-Feature) |
| `volume_chg_pct` | REAL | Volumenanstieg in % (ML-Feature) |
| `btc_trend_pct` | REAL | BTC 4h-Trend beim Einstieg (ML-Feature) |
| `price_chg_4h` | REAL | Kursanstieg 4h beim Einstieg (ML-Feature) |
| `highest_price` | REAL | Höchstkurs seit Einstieg (für Trailing Stop) |
| `trailing_active` | INTEGER | 0 = inaktiv, 1 = aktiv |
| `stop_loss_price` | REAL | Aktueller harter Stop-Loss Preis |
| `trailing_stop_price` | REAL | Aktuelles Trailing-Stop-Niveau |

Die Datenbank kann mit jedem SQLite-Browser (z.B. [DB Browser for SQLite](https://sqlitebrowser.org/)) eingesehen werden.

---

## VPS-Deployment

Für einen dauerhaft laufenden Bot auf einem Linux-Server empfiehlt sich `systemd`:

**1. Service-Datei erstellen** `/etc/systemd/system/momentumbot.service`:

```ini
[Unit]
Description=Hyperliquid Momentum Paper-Trading Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/momentumbot
ExecStart=/usr/bin/python3 /home/ubuntu/momentumbot/bot.py
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
```

**2. Service aktivieren und starten:**

```bash
sudo systemctl daemon-reload
sudo systemctl enable momentumbot
sudo systemctl start momentumbot
```

**3. Logs verfolgen:**

```bash
# systemd-Journal
sudo journalctl -u momentumbot -f

# Oder direkt das Bot-Log
tail -f /home/ubuntu/momentumbot/bot.log
```

**Wichtige Hinweise:**
- Die `.env`-Datei niemals in ein Git-Repository einchecken
- Der Bot schreibt `config.json` und `database.db` laufend – diese Dateien sollten in einem Backup-Plan berücksichtigt werden
- Bei Änderungen an `config.json` muss der Bot nicht neu gestartet werden – die Parameter werden beim nächsten Scan-Zyklus frisch geladen
