from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from contextlib import asynccontextmanager
import os
import json
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any

from data_fetcher import MultiExchangeClient
from analyzer import RMTAnalyzer
from database import Database

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs("data", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static/css", exist_ok=True)
    os.makedirs("static/js", exist_ok=True)
    app.state.db = Database("data/rmt.db")
    await app.state.db.init()
    app.state.client = MultiExchangeClient(market_type="futures")
    app.state.analyzer = RMTAnalyzer()
    yield
    await app.state.client.close()

app = FastAPI(title="RMT Crypto Trading Analysis Platform", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Dashboard loading...</h1>", status_code=200)

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/symbols")
async def get_symbols(request: Request):
    try:
        symbols = await request.app.state.client.get_exchange_info()
        if not symbols:
            symbols = await request.app.state.db.get_watchlist()
        return {"symbols": symbols[:100]}
    except Exception as e:
        return {"symbols": ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","LINKUSDT","AVAXUSDT","MATICUSDT"]}

@app.get("/api/watchlist")
async def get_watchlist(request: Request):
    symbols = await request.app.state.db.get_watchlist()
    return {"watchlist": symbols}

@app.post("/api/watchlist/{symbol}")
async def add_watchlist(symbol: str, request: Request):
    await request.app.state.db.add_to_watchlist(symbol)
    return {"success": True, "symbol": symbol.upper()}

@app.delete("/api/watchlist/{symbol}")
async def remove_watchlist(symbol: str, request: Request):
    await request.app.state.db.remove_from_watchlist(symbol)
    return {"success": True, "symbol": symbol.upper()}

@app.post("/api/market-type")
async def set_market_type(market_type: str = Query(..., description="spot or futures"), request: Request = None):
    if market_type not in ["spot", "futures"]:
        raise HTTPException(status_code=400, detail="market_type must be 'spot' or 'futures'")
    request.app.state.client.set_market_type(market_type)
    return {"market_type": market_type}

@app.post("/api/analyze")
async def analyze(
    symbol: str = Query(..., description="Trading pair e.g. BTCUSDT"),
    timeframe: str = Query(..., description="Timeframe: 5m,15m,30m,1h,2h,4h,1d"),
    market_type: str = Query("futures", description="spot or futures"),
    request: Request = None
):
    symbol = symbol.upper()
    valid_timeframes = ["5m","15m","30m","1h","2h","4h","1d"]
    if timeframe not in valid_timeframes:
        raise HTTPException(status_code=400, detail=f"Invalid timeframe. Use: {valid_timeframes}")

    request.app.state.client.set_market_type(market_type)

    try:
        candles = await request.app.state.client.get_klines(symbol, timeframe, limit=500)
        if not candles or len(candles) < 50:
            raise HTTPException(status_code=404, detail="Insufficient market data")

        tf_data = {}
        for tf in ["15m","1h","4h","1d"]:
            if tf != timeframe:
                data = await request.app.state.client.get_klines(symbol, tf, limit=200)
                if data:
                    tf_data[tf] = data

        funding = await request.app.state.client.get_funding_rate(symbol, limit=100)
        oi = await request.app.state.client.get_open_interest(symbol, limit=100)
        ls_ratio = await request.app.state.client.get_long_short_ratio(symbol, limit=100)
        stats_24h = await request.app.state.client.get_24h_stats(symbol)

        result = await request.app.state.analyzer.analyze(
            symbol=symbol,
            timeframe=timeframe,
            candles=candles,
            multi_timeframe_data=tf_data,
            funding_data=funding,
            oi_data=oi,
            ls_ratio_data=ls_ratio,
            stats_24h=stats_24h
        )

        await request.app.state.db.save_analysis(symbol, timeframe, result)

        # Include raw candles for chart rendering
        result["_candles"] = candles[-200:]  # Last 200 candles for charts

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
async def get_history(
    symbol: Optional[str] = None,
    limit: int = Query(50, ge=1, le=100),
    request: Request = None
):
    history = await request.app.state.db.get_history(symbol, limit)
    return {"history": history}

@app.post("/api/backtest")
async def run_backtest(
    symbol: str = Query(...),
    timeframe: str = Query(...),
    market_type: str = Query("futures", description="spot or futures"),
    days: int = Query(90, ge=7, le=365),
    request: Request = None
):
    symbol = symbol.upper()
    request.app.state.client.set_market_type(market_type)
    try:
        candles = await request.app.state.client.get_klines(symbol, timeframe, limit=min(days*24, 1000))
        if not candles or len(candles) < 50:
            raise HTTPException(status_code=404, detail="Insufficient data for backtest")

        result = await request.app.state.analyzer.backtest(candles, symbol, timeframe)
        await request.app.state.db.save_backtest(result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/backtests")
async def get_backtests(
    symbol: Optional[str] = None,
    limit: int = Query(20, ge=1, le=50),
    request: Request = None
):
    backtests = await request.app.state.db.get_backtests(symbol, limit)
    return {"backtests": backtests}

@app.get("/api/disclaimer")
async def get_disclaimer():
    return {
        "disclaimer": "RMT Crypto Trading Analysis Platform acts as a decision-support system rather than an auto-trading bot. Analysis runs only when the user presses Run Analysis. RMT is used to separate market noise from meaningful signals. This platform is for educational and informational purposes only. It does not constitute financial advice. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Always conduct your own research and consult a licensed financial advisor before making investment decisions. Users are solely responsible for their trading decisions.",
        "criteria": {
            "minimum_confluence": "6.0/10 for any trade consideration",
            "minimum_confidence": "65/100 for actionable signals",
            "risk_per_trade": "Maximum 1-2% account risk per setup",
            "setup_quality": "Only 'Good', 'Strong', or 'Elite' setups considered viable",
            "multi_timeframe": "Minimum 3/4 timeframes must align for directional bias",
            "liquidity_required": "Setup must have identifiable liquidity pool or stop cluster",
            "invalidation": "Every trade plan must have clear invalidation level",
            "rmt_signal": "RMT complexity score must indicate non-random market structure"
        }
    }
