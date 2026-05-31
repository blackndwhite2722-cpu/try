import aiosqlite
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

class Database:
    def __init__(self, db_path: str = "data/rmt.db"):
        self.db_path = db_path
        self.conn: Optional[aiosqlite.Connection] = None

    async def init(self):
        self.conn = await aiosqlite.connect(self.db_path)
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                result_json TEXT NOT NULL,
                confluence_score REAL,
                confidence_score REAL,
                direction TEXT
            )
        """)
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT UNIQUE NOT NULL,
                active INTEGER DEFAULT 1,
                added_at TEXT NOT NULL
            )
        """)
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS backtests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                strategy TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                total_trades INTEGER,
                win_rate REAL,
                profit_factor REAL,
                avg_rr REAL,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        await self.conn.commit()
        cursor = await self.conn.execute("SELECT COUNT(*) FROM watchlist")
        count = await cursor.fetchone()
        if count[0] == 0:
            default_coins = [
                "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
                "ADAUSDT", "DOGEUSDT", "LINKUSDT", "AVAXUSDT", "MATICUSDT"
            ]
            for coin in default_coins:
                await self.conn.execute(
                    "INSERT INTO watchlist (symbol, added_at) VALUES (?, ?)",
                    (coin, datetime.utcnow().isoformat())
                )
            await self.conn.commit()

    async def save_analysis(self, symbol: str, timeframe: str, result: Dict[str, Any]) -> int:
        timestamp = datetime.utcnow().isoformat()
        cursor = await self.conn.execute(
            """INSERT INTO analyses (symbol, timeframe, timestamp, result_json, confluence_score, confidence_score, direction)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (symbol, timeframe, timestamp, json.dumps(result),
             result.get("confluence_score", 0),
             result.get("confidence_score", 0),
             result.get("direction", "NEUTRAL"))
        )
        await self.conn.commit()
        return cursor.lastrowid

    async def get_history(self, symbol: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        if symbol:
            cursor = await self.conn.execute(
                "SELECT * FROM analyses WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
                (symbol, limit)
            )
        else:
            cursor = await self.conn.execute(
                "SELECT * FROM analyses ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "symbol": row[1],
                "timeframe": row[2],
                "timestamp": row[3],
                "result": json.loads(row[4]),
                "confluence_score": row[5],
                "confidence_score": row[6],
                "direction": row[7]
            })
        return results

    async def get_watchlist(self) -> List[str]:
        cursor = await self.conn.execute(
            "SELECT symbol FROM watchlist WHERE active = 1 ORDER BY symbol"
        )
        rows = await cursor.fetchall()
        return [row[0] for row in rows]

    async def add_to_watchlist(self, symbol: str):
        try:
            await self.conn.execute(
                "INSERT INTO watchlist (symbol, added_at) VALUES (?, ?)",
                (symbol.upper(), datetime.utcnow().isoformat())
            )
            await self.conn.commit()
        except Exception:
            pass

    async def remove_from_watchlist(self, symbol: str):
        await self.conn.execute(
            "UPDATE watchlist SET active = 0 WHERE symbol = ?",
            (symbol.upper(),)
        )
        await self.conn.commit()

    async def save_backtest(self, result: Dict[str, Any]) -> int:
        cursor = await self.conn.execute(
            """INSERT INTO backtests (symbol, timeframe, strategy, start_date, end_date, total_trades, win_rate, profit_factor, avg_rr, result_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (result["symbol"], result["timeframe"], result["strategy"],
             result["start_date"], result["end_date"], result["total_trades"],
             result["win_rate"], result["profit_factor"], result["avg_rr"],
             json.dumps(result), datetime.utcnow().isoformat())
        )
        await self.conn.commit()
        return cursor.lastrowid

    async def get_backtests(self, symbol: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        if symbol:
            cursor = await self.conn.execute(
                "SELECT * FROM backtests WHERE symbol = ? ORDER BY created_at DESC LIMIT ?",
                (symbol, limit)
            )
        else:
            cursor = await self.conn.execute(
                "SELECT * FROM backtests ORDER BY created_at DESC LIMIT ?",
                (limit,)
            )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            results.append({
                "id": row[0], "symbol": row[1], "timeframe": row[2],
                "strategy": row[3], "start_date": row[4], "end_date": row[5],
                "total_trades": row[6], "win_rate": row[7],
                "profit_factor": row[8], "avg_rr": row[9],
                "result": json.loads(row[10]), "created_at": row[11]
            })
        return results
