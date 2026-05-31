import aiohttp
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

class BaseExchangeClient:
    """Base class for exchange clients."""
    def __init__(self, name: str, base_url: str):
        self.name = name
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self._lock = asyncio.Lock()
        self._last_request_time = 0
        self._min_interval = 0.05

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def _rate_limited_request(self, endpoint: str, params: Dict = None) -> Any:
        async with self._lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self._last_request_time
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_request_time = asyncio.get_event_loop().time()

        session = await self._get_session()
        url = f"{self.base_url}{endpoint}"
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 429:
                    await asyncio.sleep(1)
                    return await self._rate_limited_request(endpoint, params)
                resp.raise_for_status()
                return await resp.json()
        except Exception as e:
            print(f"[{self.name}] Request error: {e}")
            return None

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

    async def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[Dict[str, Any]]:
        raise NotImplementedError

    async def get_ticker_price(self, symbol: str) -> float:
        raise NotImplementedError

    async def get_24h_stats(self, symbol: str) -> Dict[str, Any]:
        raise NotImplementedError

    async def get_exchange_info(self) -> List[str]:
        raise NotImplementedError


class BinanceSpotClient(BaseExchangeClient):
    """Binance Spot API client."""
    def __init__(self):
        super().__init__("Binance Spot", "https://api.binance.com")

    async def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[Dict[str, Any]]:
        data = await self._rate_limited_request(
            "/api/v3/klines",
            {"symbol": symbol.upper(), "interval": interval, "limit": limit}
        )
        if not data or not isinstance(data, list):
            return []
        candles = []
        for item in data:
            candles.append({
                "time": int(item[0]),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5]),
                "quote_volume": float(item[7]),
                "trades": int(item[8])
            })
        return candles

    async def get_ticker_price(self, symbol: str) -> float:
        data = await self._rate_limited_request(
            "/api/v3/ticker/price",
            {"symbol": symbol.upper()}
        )
        if data and "price" in data:
            return float(data["price"])
        return 0.0

    async def get_24h_stats(self, symbol: str) -> Dict[str, Any]:
        data = await self._rate_limited_request(
            "/api/v3/ticker/24hr",
            {"symbol": symbol.upper()}
        )
        if not data:
            return {}
        return {
            "price_change": float(data.get("priceChange", 0)),
            "price_change_percent": float(data.get("priceChangePercent", 0)),
            "weighted_avg_price": float(data.get("weightedAvgPrice", 0)),
            "last_price": float(data.get("lastPrice", 0)),
            "high": float(data.get("highPrice", 0)),
            "low": float(data.get("lowPrice", 0)),
            "volume": float(data.get("volume", 0)),
            "quote_volume": float(data.get("quoteVolume", 0))
        }

    async def get_exchange_info(self) -> List[str]:
        data = await self._rate_limited_request("/api/v3/exchangeInfo")
        if not data or "symbols" not in data:
            return []
        symbols = []
        for s in data["symbols"]:
            if s["status"] == "TRADING" and s["quoteAsset"] == "USDT":
                symbols.append(s["symbol"])
        return sorted(symbols)


class BinanceFuturesClient(BaseExchangeClient):
    """Binance USD-M Futures API client."""
    def __init__(self):
        super().__init__("Binance Futures", "https://fapi.binance.com")

    async def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[Dict[str, Any]]:
        data = await self._rate_limited_request(
            "/fapi/v1/klines",
            {"symbol": symbol.upper(), "interval": interval, "limit": limit}
        )
        if not data or not isinstance(data, list):
            return []
        candles = []
        for item in data:
            candles.append({
                "time": int(item[0]),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5]),
                "quote_volume": float(item[7]),
                "trades": int(item[8])
            })
        return candles

    async def get_funding_rate(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        data = await self._rate_limited_request(
            "/fapi/v1/fundingRate",
            {"symbol": symbol.upper(), "limit": limit}
        )
        if not data or not isinstance(data, list):
            return []
        return [{"time": int(d["fundingTime"]), "rate": float(d["fundingRate"])} for d in data]

    async def get_open_interest(self, symbol: str, period: str = "1h", limit: int = 100) -> List[Dict[str, Any]]:
        data = await self._rate_limited_request(
            "/fapi/v1/openInterestHist",
            {"symbol": symbol.upper(), "period": period, "limit": limit}
        )
        if not data or not isinstance(data, list):
            return []
        return [{"time": int(d["timestamp"]), "oi": float(d["sumOpenInterest"]), "value": float(d["sumOpenInterestValue"])} for d in data]

    async def get_long_short_ratio(self, symbol: str, period: str = "1h", limit: int = 100) -> List[Dict[str, Any]]:
        data = await self._rate_limited_request(
            "/fapi/v1/topLongShortAccountRatio",
            {"symbol": symbol.upper(), "period": period, "limit": limit}
        )
        if not data or not isinstance(data, list):
            return []
        return [{"time": int(d["timestamp"]), "long_ratio": float(d["longAccount"]), "short_ratio": float(d["shortAccount"]), "long_short_ratio": float(d["longShortRatio"])} for d in data]

    async def get_ticker_price(self, symbol: str) -> float:
        data = await self._rate_limited_request(
            "/fapi/v1/ticker/price",
            {"symbol": symbol.upper()}
        )
        if data and "price" in data:
            return float(data["price"])
        return 0.0

    async def get_exchange_info(self) -> List[str]:
        data = await self._rate_limited_request("/fapi/v1/exchangeInfo")
        if not data or "symbols" not in data:
            return []
        symbols = []
        for s in data["symbols"]:
            if s["status"] == "TRADING" and s["contractType"] == "PERPETUAL" and s["quoteAsset"] == "USDT":
                symbols.append(s["symbol"])
        return sorted(symbols)

    async def get_24h_stats(self, symbol: str) -> Dict[str, Any]:
        data = await self._rate_limited_request(
            "/fapi/v1/ticker/24hr",
            {"symbol": symbol.upper()}
        )
        if not data:
            return {}
        return {
            "price_change": float(data.get("priceChange", 0)),
            "price_change_percent": float(data.get("priceChangePercent", 0)),
            "weighted_avg_price": float(data.get("weightedAvgPrice", 0)),
            "last_price": float(data.get("lastPrice", 0)),
            "high": float(data.get("highPrice", 0)),
            "low": float(data.get("lowPrice", 0)),
            "volume": float(data.get("volume", 0)),
            "quote_volume": float(data.get("quoteVolume", 0))
        }


class MultiExchangeClient:
    """Unified client supporting both Spot and Futures across multiple exchanges."""
    def __init__(self, market_type: str = "futures"):
        self.market_type = market_type
        self.spot_client = BinanceSpotClient()
        self.futures_client = BinanceFuturesClient()
        self._current_client = self.futures_client if market_type == "futures" else self.spot_client

    def set_market_type(self, market_type: str):
        self.market_type = market_type
        self._current_client = self.futures_client if market_type == "futures" else self.spot_client

    async def close(self):
        await self.spot_client.close()
        await self.futures_client.close()

    async def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List[Dict[str, Any]]:
        return await self._current_client.get_klines(symbol, interval, limit)

    async def get_funding_rate(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        if self.market_type == "spot":
            return []
        return await self.futures_client.get_funding_rate(symbol, limit)

    async def get_open_interest(self, symbol: str, period: str = "1h", limit: int = 100) -> List[Dict[str, Any]]:
        if self.market_type == "spot":
            return []
        return await self.futures_client.get_open_interest(symbol, period, limit)

    async def get_long_short_ratio(self, symbol: str, period: str = "1h", limit: int = 100) -> List[Dict[str, Any]]:
        if self.market_type == "spot":
            return []
        return await self.futures_client.get_long_short_ratio(symbol, period, limit)

    async def get_ticker_price(self, symbol: str) -> float:
        return await self._current_client.get_ticker_price(symbol)

    async def get_exchange_info(self) -> List[str]:
        spot_symbols = await self.spot_client.get_exchange_info()
        futures_symbols = await self.futures_client.get_exchange_info()
        all_symbols = list(set(spot_symbols + futures_symbols))
        return sorted(all_symbols)

    async def get_24h_stats(self, symbol: str) -> Dict[str, Any]:
        return await self._current_client.get_24h_stats(symbol)
