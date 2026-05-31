import numpy as np
import math
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

class RMTAnalyzer:
    def __init__(self):
        self.layer_weights = {
            "trend_range": 0.05,
            "market_structure": 0.08,
            "liquidity": 0.07,
            "order_blocks": 0.06,
            "fvg": 0.05,
            "premium_discount": 0.04,
            "price_action": 0.07,
            "candlestick": 0.05,
            "chart_pattern": 0.06,
            "volume": 0.05,
            "open_interest": 0.04,
            "funding_rate": 0.04,
            "long_short_ratio": 0.03,
            "session": 0.03,
            "multi_timeframe": 0.08,
            "market_regime": 0.05,
            "volatility": 0.05,
            "rmt": 0.10
        }

    # ==================== MATH HELPERS ====================
    def _sma(self, data: List[float], period: int) -> List[float]:
        if len(data) < period:
            return data[:]
        result = []
        for i in range(len(data)):
            if i < period - 1:
                result.append(sum(data[:i+1]) / (i+1))
            else:
                result.append(sum(data[i-period+1:i+1]) / period)
        return result

    def _ema(self, data: List[float], period: int) -> List[float]:
        if len(data) < period:
            return data[:]
        k = 2 / (period + 1)
        result = [data[0]]
        for i in range(1, len(data)):
            result.append(data[i] * k + result[-1] * (1 - k))
        return result

    def _atr(self, candles: List[Dict], period: int = 14) -> List[float]:
        if len(candles) < period + 1:
            return [0.0] * len(candles)
        tr_list = []
        for i in range(len(candles)):
            if i == 0:
                tr = candles[i]["high"] - candles[i]["low"]
            else:
                tr1 = candles[i]["high"] - candles[i]["low"]
                tr2 = abs(candles[i]["high"] - candles[i-1]["close"])
                tr3 = abs(candles[i]["low"] - candles[i-1]["close"])
                tr = max(tr1, tr2, tr3)
            tr_list.append(tr)
        return self._ema(tr_list, period)

    def _swing_highs_lows(self, candles: List[Dict], lookback: int = 5) -> Tuple[List[int], List[int]]:
        highs, lows = [], []
        for i in range(lookback, len(candles) - lookback):
            is_high = all(candles[i]["high"] >= candles[j]["high"] for j in range(i-lookback, i+lookback+1) if j != i)
            is_low = all(candles[i]["low"] <= candles[j]["low"] for j in range(i-lookback, i+lookback+1) if j != i)
            if is_high:
                highs.append(i)
            if is_low:
                lows.append(i)
        return highs, lows

    def _std(self, data: List[float]) -> float:
        if len(data) < 2:
            return 0.0
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        return math.sqrt(variance)

    def _linear_regression(self, x: List[float], y: List[float]) -> Tuple[float, float]:
        n = len(x)
        if n == 0:
            return 0, 0
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
        if denominator == 0:
            return 0, mean_y
        slope = numerator / denominator
        intercept = mean_y - slope * mean_x
        return slope, intercept

    def _returns(self, prices: List[float]) -> List[float]:
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] == 0:
                returns.append(0.0)
            else:
                returns.append(math.log(prices[i] / prices[i-1]))
        return returns

    def _normalize(self, value: float, min_val: float, max_val: float) -> float:
        if max_val == min_val:
            return 0.5
        return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))

    # ==================== LAYER 1: TREND & RANGE ENGINE ====================
    def _layer1_trend_range(self, candles: List[Dict]) -> Dict[str, Any]:
        closes = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]

        if len(closes) < 50:
            return {"trend": "NEUTRAL", "strength": 0, "range_high": 0, "range_low": 0, "breakout_prob": 0, "direction": "NEUTRAL", "score": 0}

        ema20 = self._ema(closes, 20)
        ema50 = self._ema(closes, 50)

        # Trend detection
        last_close = closes[-1]
        trend = "NEUTRAL"
        if ema20[-1] > ema50[-1] and last_close > ema20[-1]:
            trend = "UPTREND"
        elif ema20[-1] < ema50[-1] and last_close < ema20[-1]:
            trend = "DOWNTREND"

        # Trend strength via ADX-like measure
        dm_plus = [highs[i] - highs[i-1] if highs[i] - highs[i-1] > lows[i-1] - lows[i] else 0 for i in range(1, len(highs))]
        dm_minus = [lows[i-1] - lows[i] if lows[i-1] - lows[i] > highs[i] - highs[i-1] else 0 for i in range(1, len(lows))]
        atr = self._atr(candles, 14)

        if len(atr) > 14 and atr[-1] > 0:
            di_plus = 100 * sum(dm_plus[-14:]) / (sum(atr[-14:]) + 1e-9)
            di_minus = 100 * sum(dm_minus[-14:]) / (sum(atr[-14:]) + 1e-9)
            dx = abs(di_plus - di_minus) / (di_plus + di_minus + 1e-9) * 100
            strength = min(100, dx)
        else:
            strength = 0

        # Range detection
        swing_highs, swing_lows = self._swing_highs_lows(candles, 5)
        recent_highs = [highs[i] for i in swing_highs[-10:]] if swing_highs else highs[-20:]
        recent_lows = [lows[i] for i in swing_lows[-10:]] if swing_lows else lows[-20:]
        range_high = max(recent_highs) if recent_highs else max(highs[-50:])
        range_low = min(recent_lows) if recent_lows else min(lows[-50:])

        # Breakout probability
        range_size = range_high - range_low
        current_pos = (last_close - range_low) / (range_size + 1e-9)
        volatility = self._std(closes[-20:]) / (sum(closes[-20:]) / 20 + 1e-9)
        breakout_prob = min(100, volatility * 100 * (1 - abs(current_pos - 0.5) * 2))

        # Directional bias
        direction = "BULLISH" if trend == "UPTREND" else "BEARISH" if trend == "DOWNTREND" else "NEUTRAL"
        if trend == "NEUTRAL" and breakout_prob > 60:
            direction = "BULLISH" if current_pos > 0.6 else "BEARISH" if current_pos < 0.4 else "NEUTRAL"

        score = self._normalize(strength, 0, 50) * 0.5 + self._normalize(breakout_prob, 0, 80) * 0.5

        return {
            "trend": trend,
            "strength": round(strength, 2),
            "range_high": round(range_high, 2),
            "range_low": round(range_low, 2),
            "breakout_probability": round(breakout_prob, 2),
            "direction": direction,
            "score": round(score, 2),
            "explanation": f"Market is in {trend.lower()} with {strength:.1f}% strength. Range: {range_low:.2f}-{range_high:.2f}. Breakout probability: {breakout_prob:.1f}%."
        }

    # ==================== LAYER 2: MARKET STRUCTURE ====================
    def _layer2_market_structure(self, candles: List[Dict]) -> Dict[str, Any]:
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        closes = [c["close"] for c in candles]

        if len(candles) < 30:
            return {"structure": "UNDEFINED", "hh_hl_lh_ll": [], "bos": [], "choch": [], "shifts": 0, "score": 0}

        swing_highs, swing_lows = self._swing_highs_lows(candles, 3)

        hh_hl_lh_ll = []
        bos = []
        choch = []
        structure = "UNDEFINED"

        if len(swing_highs) >= 3 and len(swing_lows) >= 3:
            # Detect HH, HL, LH, LL
            for i in range(1, min(len(swing_highs), len(swing_lows))):
                if i < len(swing_highs) and i-1 < len(swing_highs):
                    if highs[swing_highs[i]] > highs[swing_highs[i-1]]:
                        hh_hl_lh_ll.append("HH")
                    else:
                        hh_hl_lh_ll.append("LH")
                if i < len(swing_lows) and i-1 < len(swing_lows):
                    if lows[swing_lows[i]] > lows[swing_lows[i-1]]:
                        hh_hl_lh_ll.append("HL")
                    else:
                        hh_hl_lh_ll.append("LL")

            # BOS: Break of Structure
            recent_highs = swing_highs[-5:]
            recent_lows = swing_lows[-5:]
            if len(recent_highs) >= 2 and closes[-1] > highs[recent_highs[-2]]:
                bos.append({"type": "BULLISH_BOS", "level": highs[recent_highs[-2]], "index": recent_highs[-2]})
                structure = "BULLISH"
            if len(recent_lows) >= 2 and closes[-1] < lows[recent_lows[-2]]:
                bos.append({"type": "BEARISH_BOS", "level": lows[recent_lows[-2]], "index": recent_lows[-2]})
                structure = "BEARISH"

            # CHoCH: Change of Character
            if len(recent_highs) >= 3 and lows[recent_lows[-1]] < lows[recent_lows[-2]] and structure == "BULLISH":
                choch.append({"type": "BEARISH_CHOCH", "level": lows[recent_lows[-2]]})
                structure = "BEARISH_TRANSITION"
            if len(recent_lows) >= 3 and highs[recent_highs[-1]] > highs[recent_highs[-2]] and structure == "BEARISH":
                choch.append({"type": "BULLISH_CHOCH", "level": highs[recent_highs[-2]]})
                structure = "BULLISH_TRANSITION"

        # Count structure shifts
        shifts = len(bos) + len(choch)

        # Score based on structure clarity
        score = 0.5
        if structure in ["BULLISH", "BEARISH"]:
            score = 0.8
        elif structure in ["BULLISH_TRANSITION", "BEARISH_TRANSITION"]:
            score = 0.6
        if len(bos) > 0:
            score += 0.1
        score = min(1.0, score)

        return {
            "structure": structure,
            "hh_hl_lh_ll": hh_hl_lh_ll[-10:],
            "bos": bos,
            "choch": choch,
            "shifts": shifts,
            "score": round(score, 2),
            "explanation": f"Market structure: {structure}. {len(bos)} BOS events, {len(choch)} CHoCH events detected."
        }

    # ==================== LAYER 3: LIQUIDITY ENGINE ====================
    def _layer3_liquidity(self, candles: List[Dict]) -> Dict[str, Any]:
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]

        if len(candles) < 20:
            return {"pools": [], "sweeps": [], "score": 0}

        # Find equal highs/lows (liquidity pools), stop clusters, and resting liquidity clusters
        pools = []
        resting_clusters = []
        stop_clusters = []
        tolerance = self._std(highs[-50:]) * 0.5 if len(highs) >= 50 else (max(highs) - min(highs)) * 0.005

        # Detect stop clusters: areas where multiple wicks cluster together (stop loss accumulation)
        for i in range(5, len(candles) - 5):
            window_highs = [candles[j]["high"] for j in range(i-5, i+6)]
            window_lows = [candles[j]["low"] for j in range(i-5, i+6)]
            # Cluster of highs = stop cluster above
            high_cluster = sum(1 for h in window_highs if abs(h - max(window_highs)) < tolerance)
            if high_cluster >= 3:
                stop_clusters.append({"type": "STOP_CLUSTER_HIGH", "level": max(window_highs), "strength": high_cluster})
            # Cluster of lows = stop cluster below
            low_cluster = sum(1 for l in window_lows if abs(l - min(window_lows)) < tolerance)
            if low_cluster >= 3:
                stop_clusters.append({"type": "STOP_CLUSTER_LOW", "level": min(window_lows), "strength": low_cluster})

        for i in range(len(candles) - 5):
            for j in range(i + 3, min(i + 20, len(candles))):
                if abs(highs[i] - highs[j]) < tolerance:
                    pools.append({"type": "EQUAL_HIGH", "level": highs[i], "indices": [i, j], "cluster_type": "resting_liquidity"})
                if abs(lows[i] - lows[j]) < tolerance:
                    pools.append({"type": "EQUAL_LOW", "level": lows[i], "indices": [i, j], "cluster_type": "resting_liquidity"})

        # Detect resting liquidity clusters: areas where price consolidated before moving
        for i in range(5, len(candles) - 5):
            window = candles[i-5:i+5]
            window_range = max(c["high"] for c in window) - min(c["low"] for c in window)
            avg_range = sum(c["high"] - c["low"] for c in window) / 10
            if window_range < avg_range * 1.5 and avg_range > 0:
                # This is a consolidation cluster = resting liquidity
                cluster_high = max(c["high"] for c in window)
                cluster_low = min(c["low"] for c in window)
                resting_clusters.append({
                    "type": "RESTING_CLUSTER",
                    "level": (cluster_high + cluster_low) / 2,
                    "top": cluster_high,
                    "bottom": cluster_low,
                    "indices": list(range(i-5, i+5)),
                    "cluster_type": "resting_liquidity",
                    "strength": 10 / (window_range / avg_range + 1e-9)
                })

        # Merge nearby pools
        merged_pools = []
        for pool in pools:
            found = False
            for mp in merged_pools:
                if abs(mp["level"] - pool["level"]) < tolerance * 2:
                    mp["indices"].extend(pool["indices"])
                    mp["strength"] = mp.get("strength", 1) + 1
                    found = True
                    break
            if not found:
                pool["strength"] = 1
                merged_pools.append(pool)

        # Detect sweeps (price briefly beyond liquidity then reverses)
        sweeps = []
        for pool in merged_pools[-5:]:
            level = pool["level"]
            pool_type = pool["type"]
            # Check last 5 candles for sweep
            for i in range(max(0, len(candles)-5), len(candles)):
                if pool_type == "EQUAL_HIGH" and highs[i] > level and (i == len(candles)-1 or closes[i] < level):
                    sweeps.append({"type": "HIGH_SWEEP", "level": level, "index": i})
                if pool_type == "EQUAL_LOW" and lows[i] < level and (i == len(candles)-1 or closes[i] > level):
                    sweeps.append({"type": "LOW_SWEEP", "level": level, "index": i})

        # Score based on liquidity strength and recency of sweeps
        score = min(1.0, len(merged_pools) * 0.1 + len(sweeps) * 0.2)

        return {
            "pools": merged_pools[-10:],
            "resting_clusters": resting_clusters[-10:],
            "stop_clusters": stop_clusters[-10:],
            "sweeps": sweeps,
            "score": round(score, 2),
            "explanation": f"Found {len(merged_pools)} liquidity pools, {len(resting_clusters)} resting clusters, {len(stop_clusters)} stop clusters, {len(sweeps)} recent sweeps."
        }

    # ==================== LAYER 4: ORDER BLOCKS ====================
    def _layer4_order_blocks(self, candles: List[Dict]) -> Dict[str, Any]:
        if len(candles) < 10:
            return {"blocks": [], "score": 0}

        blocks = []
        for i in range(2, len(candles) - 1):
            c0, c1, c2 = candles[i-2], candles[i-1], candles[i]

            # Bullish OB: last down candle before strong up move
            if c1["close"] < c1["open"] and c2["close"] > c2["open"] and c2["close"] > c1["high"]:
                ob = {
                    "type": "BULLISH",
                    "top": c1["high"],
                    "bottom": c1["low"],
                    "index": i-1,
                    "strength": abs(c2["close"] - c1["close"]) / (c1["high"] - c1["low"] + 1e-9),
                    "mitigated": any(c["low"] <= c1["low"] for c in candles[i:])
                }
                blocks.append(ob)

            # Bearish OB: last up candle before strong down move
            if c1["close"] > c1["open"] and c2["close"] < c2["open"] and c2["close"] < c1["low"]:
                ob = {
                    "type": "BEARISH",
                    "top": c1["high"],
                    "bottom": c1["low"],
                    "index": i-1,
                    "strength": abs(c2["close"] - c1["close"]) / (c1["high"] - c1["low"] + 1e-9),
                    "mitigated": any(c["high"] >= c1["high"] for c in candles[i:])
                }
                blocks.append(ob)

        # Freshness score: unmitigated and recent blocks are stronger
        fresh_blocks = [b for b in blocks[-20:] if not b["mitigated"]]
        score = min(1.0, len(fresh_blocks) * 0.15 + sum(b["strength"] for b in fresh_blocks) * 0.1)

        return {
            "blocks": blocks[-10:],
            "fresh_blocks": fresh_blocks,
            "score": round(score, 2),
            "explanation": f"{len(blocks)} order blocks found, {len(fresh_blocks)} fresh (unmitigated)."
        }

    # ==================== LAYER 5: FAIR VALUE GAP ENGINE (with MTF alignment) ====================
    def _layer5_fvg(self, candles: List[Dict], multi_tf_candles: Dict[str, List[Dict]] = None) -> Dict[str, Any]:
        if len(candles) < 3:
            return {"fvgs": [], "unfilled_fvgs": [], "ifvgs": [], "mtf_aligned": [], "score": 0}

        fvgs = []
        atr = self._atr(candles, 14)
        current_atr = atr[-1] if atr else 0

        for i in range(1, len(candles) - 1):
            prev, curr, next_c = candles[i-1], candles[i], candles[i+1]

            # Bullish FVG: current low > previous high
            if curr["low"] > prev["high"]:
                gap_size = curr["low"] - prev["high"]
                fvg = {
                    "type": "BULLISH",
                    "top": curr["low"],
                    "bottom": prev["high"],
                    "size": gap_size,
                    "index": i,
                    "strength": gap_size / (current_atr + 1e-9),
                    "filled": any(c["low"] <= prev["high"] for c in candles[i+1:]),
                    "ifvg": any(c["low"] <= prev["high"] and c["high"] >= curr["low"] for c in candles[i+1:])
                }
                fvgs.append(fvg)

            # Bearish FVG: current high < previous low
            if curr["high"] < prev["low"]:
                gap_size = prev["low"] - curr["high"]
                fvg = {
                    "type": "BEARISH",
                    "top": prev["low"],
                    "bottom": curr["high"],
                    "size": gap_size,
                    "index": i,
                    "strength": gap_size / (current_atr + 1e-9),
                    "filled": any(c["high"] >= prev["low"] for c in candles[i+1:]),
                    "ifvg": any(c["high"] >= prev["low"] and c["low"] <= curr["high"] for c in candles[i+1:])
                }
                fvgs.append(fvg)

        unfilled = [f for f in fvgs[-15:] if not f["filled"]]
        ifvgs = [f for f in fvgs[-15:] if f["ifvg"]]

        # Multi-timeframe alignment: check if FVGs align across timeframes
        mtf_aligned = []
        if multi_tf_candles:
            for tf, tf_candles in multi_tf_candles.items():
                if len(tf_candles) < 3:
                    continue
                for i in range(1, len(tf_candles) - 1):
                    tf_prev, tf_curr = tf_candles[i-1], tf_candles[i]
                    # Check if this TF has FVG in same price zone as primary TF FVGs
                    for fvg in unfilled:
                        zone_tolerance = (fvg["top"] - fvg["bottom"]) * 2
                        if tf_curr["low"] > tf_prev["high"]:
                            if abs(tf_curr["low"] - fvg["top"]) < zone_tolerance or abs(tf_prev["high"] - fvg["bottom"]) < zone_tolerance:
                                mtf_aligned.append({"timeframe": tf, "type": "BULLISH", "level": tf_curr["low"]})
                        elif tf_curr["high"] < tf_prev["low"]:
                            if abs(tf_curr["high"] - fvg["bottom"]) < zone_tolerance or abs(tf_prev["low"] - fvg["top"]) < zone_tolerance:
                                mtf_aligned.append({"timeframe": tf, "type": "BEARISH", "level": tf_curr["high"]})

        # Imbalance strength: average strength of unfilled FVGs
        avg_strength = sum(f["strength"] for f in unfilled) / len(unfilled) if unfilled else 0

        # Score: unfilled + MTF aligned = stronger
        base_score = min(1.0, len(unfilled) * 0.1 + avg_strength * 0.05)
        if mtf_aligned:
            base_score = min(1.0, base_score + len(mtf_aligned) * 0.05)

        return {
            "fvgs": fvgs[-10:],
            "unfilled_fvgs": unfilled,
            "ifvgs": ifvgs,
            "mtf_aligned": mtf_aligned,
            "score": round(base_score, 2),
            "explanation": f"{len(fvgs)} FVGs detected. {len(unfilled)} unfilled gaps with avg imbalance strength {avg_strength:.2f}. {len(ifvgs)} IFVGs. {len(mtf_aligned)} MTF-aligned."
        }
    # ==================== LAYER 6: PREMIUM / DISCOUNT ZONES ====================
    def _layer6_premium_discount(self, candles: List[Dict]) -> Dict[str, Any]:
        if len(candles) < 50:
            return {"equilibrium": 0, "zone": "NEUTRAL", "optimal_entry": 0, "score": 0.5}

        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        closes = [c["close"] for c in candles]

        swing_highs, swing_lows = self._swing_highs_lows(candles, 5)
        if not swing_highs or not swing_lows:
            range_high = max(highs[-50:])
            range_low = min(lows[-50:])
        else:
            range_high = max(highs[i] for i in swing_highs[-10:])
            range_low = min(lows[i] for i in swing_lows[-10:])

        equilibrium = (range_high + range_low) / 2
        last_close = closes[-1]

        # Determine zone
        if last_close > equilibrium + (range_high - range_low) * 0.1:
            zone = "PREMIUM"
        elif last_close < equilibrium - (range_high - range_low) * 0.1:
            zone = "DISCOUNT"
        else:
            zone = "EQUILIBRIUM"

        # Optimal trade entry
        # In uptrend, discount is optimal. In downtrend, premium is optimal for shorts
        optimal_entry = equilibrium  # Default

        # Score: discount in uptrend or premium in downtrend is better
        ema20 = self._ema(closes, 20)
        ema50 = self._ema(closes, 50)
        trend = "UP" if ema20[-1] > ema50[-1] else "DOWN"

        if (trend == "UP" and zone == "DISCOUNT") or (trend == "DOWN" and zone == "PREMIUM"):
            score = 0.9
        elif zone == "EQUILIBRIUM":
            score = 0.5
        else:
            score = 0.3

        return {
            "equilibrium": round(equilibrium, 2),
            "zone": zone,
            "range_high": round(range_high, 2),
            "range_low": round(range_low, 2),
            "optimal_entry": round(optimal_entry, 2),
            "score": round(score, 2),
            "explanation": f"Price in {zone} zone. Equilibrium: {equilibrium:.2f}. Range: {range_low:.2f}-{range_high:.2f}."
        }

    # ==================== LAYER 7: PRICE ACTION ENGINE ====================
    def _layer7_price_action(self, candles: List[Dict]) -> Dict[str, Any]:
        if len(candles) < 20:
            return {"sr_levels": [], "break_retests": [], "rejections": [], "momentum": "NEUTRAL", "score": 0.5}

        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        closes = [c["close"] for c in candles]
        opens = [c["open"] for c in candles]

        # Support/Resistance from swing points
        swing_highs, swing_lows = self._swing_highs_lows(candles, 3)
        sr_levels = []
        for i in swing_highs[-8:]:
            sr_levels.append({"level": highs[i], "type": "RESISTANCE", "strength": 1})
        for i in swing_lows[-8:]:
            sr_levels.append({"level": lows[i], "type": "SUPPORT", "strength": 1})

        # Merge close levels
        merged_sr = []
        tolerance = (max(highs[-50:]) - min(lows[-50:])) * 0.005
        for level in sr_levels:
            found = False
            for ml in merged_sr:
                if abs(ml["level"] - level["level"]) < tolerance:
                    ml["strength"] += 1
                    ml["type"] = "S/R" if ml["type"] != level["type"] else ml["type"]
                    found = True
                    break
            if not found:
                merged_sr.append(level)

        # Break-retest detection
        break_retests = []
        for level in merged_sr[-5:]:
            lvl = level["level"]
            for i in range(len(candles) - 5, len(candles)):
                if i > 0 and abs(closes[i-1] - lvl) < tolerance and abs(closes[i] - lvl) < tolerance:
                    break_retests.append({"level": lvl, "index": i, "type": "RETEST"})

        # Rejection candles (long wicks)
        rejections = []
        for i in range(len(candles) - 5, len(candles)):
            body = abs(closes[i] - opens[i])
            upper_wick = highs[i] - max(closes[i], opens[i])
            lower_wick = min(closes[i], opens[i]) - lows[i]
            if body > 0:
                if upper_wick > body * 2 and closes[i] < opens[i]:
                    rejections.append({"index": i, "type": "UPPER_REJECTION", "strength": upper_wick / body})
                if lower_wick > body * 2 and closes[i] > opens[i]:
                    rejections.append({"index": i, "type": "LOWER_REJECTION", "strength": lower_wick / body})

        # Momentum shift
        momentum = "NEUTRAL"
        if len(closes) >= 5:
            last_5 = closes[-5:]
            if all(last_5[i] > last_5[i-1] for i in range(1, 5)):
                momentum = "BULLISH"
            elif all(last_5[i] < last_5[i-1] for i in range(1, 5)):
                momentum = "BEARISH"

        # Compression detection: narrowing range + declining volatility
        compression = False
        compression_strength = 0
        if len(candles) >= 20:
            first_10_range = max(highs[-20:-10]) - min(lows[-20:-10])
            last_10_range = max(highs[-10:]) - min(lows[-10:])
            if first_10_range > 0:
                range_ratio = last_10_range / first_10_range
                compression = range_ratio < 0.6
                compression_strength = 1 - range_ratio if compression else 0

            # Also check ATR compression
            atr = self._atr(candles, 14)
            if len(atr) >= 20:
                atr_first = sum(atr[-20:-10]) / 10
                atr_last = sum(atr[-10:]) / 10
                if atr_first > 0 and atr_last / atr_first < 0.7:
                    compression = True
                    compression_strength = max(compression_strength, 1 - (atr_last / atr_first))

        score = min(1.0, len(merged_sr) * 0.05 + len(rejections) * 0.1 + (0.8 if momentum != "NEUTRAL" else 0.4) + (compression_strength * 0.1))

        return {
            "sr_levels": merged_sr,
            "break_retests": break_retests,
            "rejections": rejections,
            "momentum": momentum,
            "compression": compression,
            "compression_strength": round(compression_strength, 2),
            "score": round(score, 2),
            "explanation": f"{len(merged_sr)} S/R levels, {len(rejections)} rejection candles, momentum: {momentum}. {'Compression detected!' if compression else 'No compression.'}"
        }

    # ==================== LAYER 8: CANDLESTICK SCANNER (25+ Patterns) ====================
    def _layer8_candlestick(self, candles: List[Dict]) -> Dict[str, Any]:
        if len(candles) < 5:
            return {"patterns": [], "score": 0}

        patterns = []

        for i in range(2, len(candles)):
            c = candles[i]
            c1 = candles[i-1]
            c2 = candles[i-2]
            c3 = candles[i-3] if i >= 3 else c2

            body = abs(c["close"] - c["open"])
            range_c = c["high"] - c["low"]
            upper_wick = c["high"] - max(c["close"], c["open"])
            lower_wick = min(c["close"], c["open"]) - c["low"]
            body_dir = "BULLISH" if c["close"] > c["open"] else "BEARISH"
            prev_body = abs(c1["close"] - c1["open"])
            prev_dir = "BULLISH" if c1["close"] > c1["open"] else "BEARISH"

            if range_c == 0:
                continue

            body_pct = body / range_c
            upper_pct = upper_wick / range_c
            lower_pct = lower_wick / range_c

            # --- Single Candle Patterns ---

            # Doji (open ≈ close)
            if body_pct < 0.1 and range_c > 0:
                patterns.append({"index": i, "pattern": "DOJI", "strength": 0.5, "type": body_dir})

            # Dragonfly Doji (long lower wick, no upper wick)
            if body_pct < 0.1 and lower_pct > 0.6 and upper_pct < 0.1:
                patterns.append({"index": i, "pattern": "DRAGONFLY_DOJI", "strength": 0.8, "type": "BULLISH"})

            # Gravestone Doji (long upper wick, no lower wick)
            if body_pct < 0.1 and upper_pct > 0.6 and lower_pct < 0.1:
                patterns.append({"index": i, "pattern": "GRAVESTONE_DOJI", "strength": 0.8, "type": "BEARISH"})

            # Long-Legged Doji (long wicks both sides)
            if body_pct < 0.1 and upper_pct > 0.3 and lower_pct > 0.3:
                patterns.append({"index": i, "pattern": "LONG_LEGGED_DOJI", "strength": 0.6, "type": "NEUTRAL"})

            # Hammer (small body at top, long lower wick)
            if lower_wick > body * 2 and upper_wick < body * 0.5 and body_dir == "BULLISH":
                patterns.append({"index": i, "pattern": "HAMMER", "strength": 0.75, "type": "BULLISH"})

            # Inverted Hammer (small body at bottom, long upper wick)
            if upper_wick > body * 2 and lower_wick < body * 0.5 and body_dir == "BULLISH":
                patterns.append({"index": i, "pattern": "INVERTED_HAMMER", "strength": 0.7, "type": "BULLISH"})

            # Hanging Man (hammer at top of uptrend)
            if lower_wick > body * 2 and upper_wick < body * 0.5 and body_dir == "BEARISH" and i >= 5:
                recent_trend = candles[i]["close"] > candles[i-5]["close"]
                if recent_trend:
                    patterns.append({"index": i, "pattern": "HANGING_MAN", "strength": 0.8, "type": "BEARISH"})

            # Shooting Star (inverted hammer at top)
            if upper_wick > body * 2 and lower_wick < body * 0.5 and body_dir == "BEARISH" and i >= 5:
                recent_trend = candles[i]["close"] > candles[i-5]["close"]
                if recent_trend:
                    patterns.append({"index": i, "pattern": "SHOOTING_STAR", "strength": 0.8, "type": "BEARISH"})

            # Marubozu (no wicks, strong body)
            if body_pct > 0.95 and upper_pct < 0.05 and lower_pct < 0.05:
                name = "BULLISH_MARUBOZU" if body_dir == "BULLISH" else "BEARISH_MARUBOZU"
                patterns.append({"index": i, "pattern": name, "strength": 0.85, "type": body_dir})

            # Spinning Top (small body, long wicks both sides)
            if body_pct < 0.3 and upper_pct > 0.3 and lower_pct > 0.3:
                patterns.append({"index": i, "pattern": "SPINNING_TOP", "strength": 0.4, "type": "NEUTRAL"})

            # Belt Hold (open at low/high, strong body)
            if body_pct > 0.8:
                if body_dir == "BULLISH" and abs(c["open"] - c["low"]) < range_c * 0.05:
                    patterns.append({"index": i, "pattern": "BULLISH_BELT_HOLD", "strength": 0.75, "type": "BULLISH"})
                if body_dir == "BEARISH" and abs(c["open"] - c["high"]) < range_c * 0.05:
                    patterns.append({"index": i, "pattern": "BEARISH_BELT_HOLD", "strength": 0.75, "type": "BEARISH"})

            # --- Two Candle Patterns ---

            # Engulfing
            if prev_body > 0 and body > prev_body * 1.2:
                if body_dir == "BULLISH" and prev_dir == "BEARISH" and c["open"] < c1["close"] and c["close"] > c1["open"]:
                    patterns.append({"index": i, "pattern": "BULLISH_ENGULFING", "strength": 0.85, "type": "BULLISH"})
                if body_dir == "BEARISH" and prev_dir == "BULLISH" and c["open"] > c1["close"] and c["close"] < c1["open"]:
                    patterns.append({"index": i, "pattern": "BEARISH_ENGULFING", "strength": 0.85, "type": "BEARISH"})

            # Tweezer Top (two candles with same high, first bullish second bearish)
            if abs(c["high"] - c1["high"]) < range_c * 0.05 and prev_dir == "BULLISH" and body_dir == "BEARISH":
                patterns.append({"index": i, "pattern": "TWEEZER_TOP", "strength": 0.75, "type": "BEARISH"})

            # Tweezer Bottom (two candles with same low, first bearish second bullish)
            if abs(c["low"] - c1["low"]) < range_c * 0.05 and prev_dir == "BEARISH" and body_dir == "BULLISH":
                patterns.append({"index": i, "pattern": "TWEEZER_BOTTOM", "strength": 0.75, "type": "BULLISH"})

            # Piercing Line (bullish, closes past midpoint of prev bearish candle)
            if prev_dir == "BEARISH" and body_dir == "BULLISH":
                midpoint = (c1["open"] + c1["close"]) / 2
                if c["close"] > midpoint and c["open"] < c1["low"]:
                    patterns.append({"index": i, "pattern": "PIERCING_LINE", "strength": 0.8, "type": "BULLISH"})

            # Dark Cloud Cover (bearish, closes below midpoint of prev bullish candle)
            if prev_dir == "BULLISH" and body_dir == "BEARISH":
                midpoint = (c1["open"] + c1["close"]) / 2
                if c["close"] < midpoint and c["open"] > c1["high"]:
                    patterns.append({"index": i, "pattern": "DARK_CLOUD_COVER", "strength": 0.8, "type": "BEARISH"})

            # Counterattack (same close as prev opposite candle)
            if prev_dir != body_dir and abs(c["close"] - c1["close"]) < range_c * 0.05 and prev_body > 0:
                name = "BULLISH_COUNTERATTACK" if body_dir == "BULLISH" else "BEARISH_COUNTERATTACK"
                patterns.append({"index": i, "pattern": name, "strength": 0.7, "type": body_dir})

            # Kicking (gap in opposite direction, marubozu)
            if body_pct > 0.9 and prev_body > 0:
                if prev_dir == "BEARISH" and body_dir == "BULLISH" and c["open"] > c1["close"]:
                    patterns.append({"index": i, "pattern": "BULLISH_KICKING", "strength": 0.85, "type": "BULLISH"})
                if prev_dir == "BULLISH" and body_dir == "BEARISH" and c["open"] < c1["close"]:
                    patterns.append({"index": i, "pattern": "BEARISH_KICKING", "strength": 0.85, "type": "BEARISH"})

            # Meeting Lines (opposite candles, same close)
            if prev_dir != body_dir and abs(c["close"] - c1["close"]) < range_c * 0.03 and prev_body > 0 and body > 0:
                name = "BULLISH_MEETING_LINES" if body_dir == "BULLISH" else "BEARISH_MEETING_LINES"
                patterns.append({"index": i, "pattern": name, "strength": 0.65, "type": body_dir})

            # Separating Lines (opposite candles, same open)
            if prev_dir != body_dir and abs(c["open"] - c1["open"]) < range_c * 0.03 and prev_body > 0 and body > 0:
                name = "BULLISH_SEPARATING_LINES" if body_dir == "BULLISH" else "BEARISH_SEPARATING_LINES"
                patterns.append({"index": i, "pattern": name, "strength": 0.65, "type": body_dir})

            # --- Three Candle Patterns ---

            # Morning Star
            if i >= 2:
                c2_body = abs(c2["close"] - c2["open"])
                c2_dir = "BULLISH" if c2["close"] > c2["open"] else "BEARISH"
                c1_body = abs(c1["close"] - c1["open"])
                if c2_dir == "BEARISH" and c1_body < c2_body * 0.3 and body_dir == "BULLISH" and c["close"] > (c2["open"] + c2["close"]) / 2:
                    patterns.append({"index": i, "pattern": "MORNING_STAR", "strength": 0.9, "type": "BULLISH"})

            # Evening Star
            if i >= 2:
                c2_body = abs(c2["close"] - c2["open"])
                c2_dir = "BULLISH" if c2["close"] > c2["open"] else "BEARISH"
                c1_body = abs(c1["close"] - c1["open"])
                if c2_dir == "BULLISH" and c1_body < c2_body * 0.3 and body_dir == "BEARISH" and c["close"] < (c2["open"] + c2["close"]) / 2:
                    patterns.append({"index": i, "pattern": "EVENING_STAR", "strength": 0.9, "type": "BEARISH"})

            # Three White Soldiers
            if i >= 2 and c["close"] > c["open"] and c1["close"] > c1["open"] and c2["close"] > c2["open"]:
                if c["close"] > c1["close"] > c2["close"] and body > 0 and c["open"] > c1["open"] > c2["open"]:
                    patterns.append({"index": i, "pattern": "THREE_WHITE_SOLDIERS", "strength": 0.9, "type": "BULLISH"})

            # Three Black Crows
            if i >= 2 and c["close"] < c["open"] and c1["close"] < c1["open"] and c2["close"] < c2["open"]:
                if c["close"] < c1["close"] < c2["close"] and body > 0 and c["open"] < c1["open"] < c2["open"]:
                    patterns.append({"index": i, "pattern": "THREE_BLACK_CROWS", "strength": 0.9, "type": "BEARISH"})

            # Harami
            if i > 0 and prev_body > 0:
                if c["high"] < max(c1["open"], c1["close"]) and c["low"] > min(c1["open"], c1["close"]) and body < prev_body * 0.7:
                    if prev_dir == "BEARISH":
                        patterns.append({"index": i, "pattern": "BULLISH_HARAMI", "strength": 0.7, "type": "BULLISH"})
                    else:
                        patterns.append({"index": i, "pattern": "BEARISH_HARAMI", "strength": 0.7, "type": "BEARISH"})

            # Harami Cross (harami with doji inside)
            if i > 0 and prev_body > 0 and body_pct < 0.1:
                if c["high"] < max(c1["open"], c1["close"]) and c["low"] > min(c1["open"], c1["close"]):
                    if prev_dir == "BEARISH":
                        patterns.append({"index": i, "pattern": "BULLISH_HARAMI_CROSS", "strength": 0.8, "type": "BULLISH"})
                    else:
                        patterns.append({"index": i, "pattern": "BEARISH_HARAMI_CROSS", "strength": 0.8, "type": "BEARISH"})

            # Advance Block (3 bullish candles with weakening)
            if i >= 2 and c["close"] > c["open"] and c1["close"] > c1["open"] and c2["close"] > c2["open"]:
                if upper_wick > body * 0.3 and c["close"] - c["open"] < c1["close"] - c1["open"]:
                    patterns.append({"index": i, "pattern": "ADVANCE_BLOCK", "strength": 0.6, "type": "BEARISH"})

            # Stalled Pattern (3 bullish, last small with long upper wick)
            if i >= 2 and c["close"] > c["open"] and c1["close"] > c1["open"] and c2["close"] > c2["open"]:
                if body < prev_body * 0.5 and upper_wick > body * 1.5:
                    patterns.append({"index": i, "pattern": "STALLED_PATTERN", "strength": 0.6, "type": "BEARISH"})

            # Deliberation (3 bullish, last doji/spinning top)
            if i >= 2 and c2["close"] > c2["open"] and c1["close"] > c1["open"] and body_pct < 0.3:
                patterns.append({"index": i, "pattern": "DELIBERATION", "strength": 0.55, "type": "BEARISH"})

            # Mat Hold (gap up, small pullback, then continuation)
            if i >= 3:
                if c3["close"] > c3["open"] and c2["close"] < c2["open"] and c1["close"] < c1["open"] and body_dir == "BULLISH":
                    if c["close"] > c3["close"] and c2["open"] > c3["close"]:
                        patterns.append({"index": i, "pattern": "BULLISH_MAT_HOLD", "strength": 0.8, "type": "BULLISH"})

            # Rising Three Methods (bullish continuation)
            if i >= 4:
                c4 = candles[i-4]
                if c4["close"] > c4["open"] and c["close"] > c["open"]:
                    mid_candles = candles[i-3:i]
                    all_small = all(abs(mc["close"] - mc["open"]) < abs(c4["close"] - c4["open"]) * 0.5 for mc in mid_candles)
                    if all_small and c["close"] > c4["close"]:
                        patterns.append({"index": i, "pattern": "RISING_THREE_METHODS", "strength": 0.85, "type": "BULLISH"})

            # Falling Three Methods (bearish continuation)
            if i >= 4:
                c4 = candles[i-4]
                if c4["close"] < c4["open"] and c["close"] < c["open"]:
                    mid_candles = candles[i-3:i]
                    all_small = all(abs(mc["close"] - mc["open"]) < abs(c4["close"] - c4["open"]) * 0.5 for mc in mid_candles)
                    if all_small and c["close"] < c4["close"]:
                        patterns.append({"index": i, "pattern": "FALLING_THREE_METHODS", "strength": 0.85, "type": "BEARISH"})

            # Stick Sandwich (bearish-bullish-bearish with same lows)
            if i >= 2:
                if c2["close"] < c2["open"] and c1["close"] > c1["open"] and c["close"] < c["open"]:
                    if abs(c2["low"] - c["low"]) < (c2["high"] - c2["low"]) * 0.05 and c1["close"] < c2["open"]:
                        patterns.append({"index": i, "pattern": "STICK_SANDWICH", "strength": 0.7, "type": "BULLISH"})

            # Ladder Bottom (5 candles, 4 bearish then bullish gap up)
            if i >= 4:
                prev4 = candles[i-4:i]
                if all(p["close"] < p["open"] for p in prev4) and body_dir == "BULLISH":
                    if c["open"] > c1["close"]:
                        patterns.append({"index": i, "pattern": "LADDER_BOTTOM", "strength": 0.85, "type": "BULLISH"})

            # Window (Gap)
            if i > 0:
                if body_dir == "BULLISH" and c["low"] > c1["high"]:
                    patterns.append({"index": i, "pattern": "BULLISH_WINDOW", "strength": 0.75, "type": "BULLISH"})
                if body_dir == "BEARISH" and c["high"] < c1["low"]:
                    patterns.append({"index": i, "pattern": "BEARISH_WINDOW", "strength": 0.75, "type": "BEARISH"})

            # On Neck Line
            if i > 0 and prev_dir == "BEARISH" and body_dir == "BULLISH":
                if abs(c["close"] - c1["low"]) < range_c * 0.02:
                    patterns.append({"index": i, "pattern": "ON_NECK_LINE", "strength": 0.55, "type": "BEARISH"})

            # In Neck Line
            if i > 0 and prev_dir == "BEARISH" and body_dir == "BULLISH":
                if c["close"] > c1["low"] and c["close"] < (c1["low"] + c1["close"]) / 2:
                    patterns.append({"index": i, "pattern": "IN_NECK_LINE", "strength": 0.55, "type": "BEARISH"})

            # Thrusting Line
            if i > 0 and prev_dir == "BEARISH" and body_dir == "BULLISH":
                midpoint = (c1["open"] + c1["close"]) / 2
                if c["close"] > midpoint and c["close"] < c1["open"]:
                    patterns.append({"index": i, "pattern": "THRUSTING_LINE", "strength": 0.6, "type": "BEARISH"})

        recent_patterns = [p for p in patterns if p["index"] >= len(candles) - 5]
        bullish = [p for p in recent_patterns if p["type"] == "BULLISH"]
        bearish = [p for p in recent_patterns if p["type"] == "BEARISH"]
        neutral = [p for p in recent_patterns if p["type"] == "NEUTRAL"]

        score = min(1.0, len(recent_patterns) * 0.08 + sum(p["strength"] for p in recent_patterns) * 0.05)

        return {
            "patterns": patterns[-15:],
            "recent_patterns": recent_patterns,
            "bullish_count": len(bullish),
            "bearish_count": len(bearish),
            "neutral_count": len(neutral),
            "score": round(score, 2),
            "explanation": f"{len(recent_patterns)} recent patterns: {len(bullish)} bullish, {len(bearish)} bearish, {len(neutral)} neutral."
        }

    # ==================== LAYER 9: CHART PATTERN SCANNER (20+ Patterns) ====================
    def _layer9_chart_patterns(self, candles: List[Dict]) -> Dict[str, Any]:
        if len(candles) < 30:
            return {"patterns": [], "score": 0}

        patterns = []
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        closes = [c["close"] for c in candles]
        opens = [c["open"] for c in candles]

        swing_highs, swing_lows = self._swing_highs_lows(candles, 5)

        # --- Triangle Patterns ---
        if len(swing_highs) >= 3 and len(swing_lows) >= 3:
            recent_sh = swing_highs[-5:]
            recent_sl = swing_lows[-5:]

            if len(recent_sh) >= 2 and len(recent_sl) >= 2:
                x_highs = list(range(len(recent_sh)))
                y_highs = [highs[i] for i in recent_sh]
                slope_high, _ = self._linear_regression(x_highs, y_highs)

                x_lows = list(range(len(recent_sl)))
                y_lows = [lows[i] for i in recent_sl]
                slope_low, _ = self._linear_regression(x_lows, y_lows)

                if slope_high < -0.001 and slope_low > 0.001:
                    patterns.append({"pattern": "SYMMETRICAL_TRIANGLE", "strength": 0.6, "direction": "NEUTRAL"})
                elif slope_high < -0.001 and abs(slope_low) < 0.001:
                    patterns.append({"pattern": "DESCENDING_TRIANGLE", "strength": 0.6, "direction": "BEARISH"})
                elif abs(slope_high) < 0.001 and slope_low > 0.001:
                    patterns.append({"pattern": "ASCENDING_TRIANGLE", "strength": 0.6, "direction": "BULLISH"})

        # --- Head & Shoulders ---
        if len(swing_highs) >= 3:
            for i in range(len(swing_highs) - 2):
                left, head, right = swing_highs[i], swing_highs[i+1], swing_highs[i+2]
                if highs[head] > highs[left] and highs[head] > highs[right] and abs(highs[left] - highs[right]) < (highs[head] - highs[left]) * 0.3:
                    neckline = min(lows[left:right+1])
                    patterns.append({"pattern": "HEAD_AND_SHOULDERS", "strength": 0.8, "direction": "BEARISH", "neckline": neckline})

        # Inverse H&S
        if len(swing_lows) >= 3:
            for i in range(len(swing_lows) - 2):
                left, head, right = swing_lows[i], swing_lows[i+1], swing_lows[i+2]
                if lows[head] < lows[left] and lows[head] < lows[right] and abs(lows[left] - lows[right]) < (lows[left] - lows[head]) * 0.3:
                    neckline = max(highs[left:right+1])
                    patterns.append({"pattern": "INVERSE_HEAD_AND_SHOULDERS", "strength": 0.8, "direction": "BULLISH", "neckline": neckline})

        # --- Double Top / Bottom ---
        if len(swing_highs) >= 2:
            for i in range(len(swing_highs) - 1):
                if abs(highs[swing_highs[i]] - highs[swing_highs[i+1]]) < (max(highs) - min(highs)) * 0.02:
                    patterns.append({"pattern": "DOUBLE_TOP", "strength": 0.7, "direction": "BEARISH", "level": highs[swing_highs[i]]})

        if len(swing_lows) >= 2:
            for i in range(len(swing_lows) - 1):
                if abs(lows[swing_lows[i]] - lows[swing_lows[i+1]]) < (max(highs) - min(lows)) * 0.02:
                    patterns.append({"pattern": "DOUBLE_BOTTOM", "strength": 0.7, "direction": "BULLISH", "level": lows[swing_lows[i]]})

        # --- Triple Top / Bottom ---
        if len(swing_highs) >= 3:
            for i in range(len(swing_highs) - 2):
                h1, h2, h3 = highs[swing_highs[i]], highs[swing_highs[i+1]], highs[swing_highs[i+2]]
                if abs(h1 - h2) < (max(highs) - min(highs)) * 0.02 and abs(h2 - h3) < (max(highs) - min(highs)) * 0.02:
                    patterns.append({"pattern": "TRIPLE_TOP", "strength": 0.8, "direction": "BEARISH", "level": h1})

        if len(swing_lows) >= 3:
            for i in range(len(swing_lows) - 2):
                l1, l2, l3 = lows[swing_lows[i]], lows[swing_lows[i+1]], lows[swing_lows[i+2]]
                if abs(l1 - l2) < (max(highs) - min(lows)) * 0.02 and abs(l2 - l3) < (max(highs) - min(lows)) * 0.02:
                    patterns.append({"pattern": "TRIPLE_BOTTOM", "strength": 0.8, "direction": "BULLISH", "level": l1})

        # --- Flag / Pennant ---
        if len(candles) >= 20:
            first_10_range = max(highs[:10]) - min(lows[:10])
            last_10_range = max(highs[-10:]) - min(lows[-10:])
            if first_10_range > last_10_range * 3 and last_10_range > 0:
                if closes[-1] > closes[-10]:
                    patterns.append({"pattern": "BULL_FLAG", "strength": 0.6, "direction": "BULLISH"})
                else:
                    patterns.append({"pattern": "BEAR_FLAG", "strength": 0.6, "direction": "BEARISH"})

            # Pennant (smaller consolidation than flag)
            if first_10_range > last_10_range * 4 and last_10_range > 0 and len(candles) >= 15:
                mid_highs = [highs[i] for i in range(len(candles)-15, len(candles))]
                mid_lows = [lows[i] for i in range(len(candles)-15, len(candles))]
                if max(mid_highs) - min(mid_lows) < first_10_range * 0.15:
                    if closes[-1] > closes[-15]:
                        patterns.append({"pattern": "BULL_PENNANT", "strength": 0.65, "direction": "BULLISH"})
                    else:
                        patterns.append({"pattern": "BEAR_PENNANT", "strength": 0.65, "direction": "BEARISH"})

        # --- Wedge Patterns ---
        if len(swing_highs) >= 3 and len(swing_lows) >= 3:
            recent_sh = swing_highs[-4:]
            recent_sl = swing_lows[-4:]
            if len(recent_sh) >= 2 and len(recent_sl) >= 2:
                x = list(range(len(recent_sh)))
                y_h = [highs[i] for i in recent_sh]
                y_l = [lows[i] for i in recent_sl]
                s_h, _ = self._linear_regression(x, y_h)
                s_l, _ = self._linear_regression(x, y_l)
                if s_h < -0.001 and s_l > 0.001 and abs(s_h) > 0.001 and abs(s_l) > 0.001:
                    patterns.append({"pattern": "FALLING_WEDGE", "strength": 0.6, "direction": "BULLISH"})
                if s_h > 0.001 and s_l < -0.001:
                    patterns.append({"pattern": "RISING_WEDGE", "strength": 0.6, "direction": "BEARISH"})

        # --- Cup and Handle ---
        if len(candles) >= 50:
            mid = len(candles) // 2
            cup_depth = max(highs[:mid]) - min(lows[:mid])
            handle_depth = max(highs[mid:mid+10]) - min(lows[mid:mid+10])
            if cup_depth > 0 and handle_depth < cup_depth * 0.3 and closes[-1] > (max(highs[:mid]) + min(lows[:mid])) / 2:
                patterns.append({"pattern": "CUP_AND_HANDLE", "strength": 0.5, "direction": "BULLISH"})

        # --- Inverted Cup and Handle ---
        if len(candles) >= 50:
            mid = len(candles) // 2
            cup_depth = max(highs[:mid]) - min(lows[:mid])
            handle_depth = max(highs[mid:mid+10]) - min(lows[mid:mid+10])
            if cup_depth > 0 and handle_depth < cup_depth * 0.3 and closes[-1] < (max(highs[:mid]) + min(lows[:mid])) / 2:
                patterns.append({"pattern": "INVERTED_CUP_AND_HANDLE", "strength": 0.5, "direction": "BEARISH"})

        # --- Rounding Bottom / Top ---
        if len(candles) >= 40:
            first_20 = candles[:20]
            last_20 = candles[-20:]
            mid_20 = candles[len(candles)//2 - 10:len(candles)//2 + 10]
            first_avg = sum(c["close"] for c in first_20) / 20
            mid_avg = sum(c["close"] for c in mid_20) / 20
            last_avg = sum(c["close"] for c in last_20) / 20

            if first_avg > mid_avg and last_avg > mid_avg and mid_avg < first_avg * 0.98:
                patterns.append({"pattern": "ROUNDING_BOTTOM", "strength": 0.55, "direction": "BULLISH"})
            if first_avg < mid_avg and last_avg < mid_avg and mid_avg > first_avg * 1.02:
                patterns.append({"pattern": "ROUNDING_TOP", "strength": 0.55, "direction": "BEARISH"})

        # --- Rectangle ---
        if len(candles) >= 20:
            recent_highs = highs[-20:]
            recent_lows = lows[-20:]
            range_size = max(recent_highs) - min(recent_lows)
            if range_size > 0:
                touches_high = sum(1 for h in recent_highs if h > max(recent_highs) - range_size * 0.02)
                touches_low = sum(1 for l in recent_lows if l < min(recent_lows) + range_size * 0.02)
                if touches_high >= 2 and touches_low >= 2 and range_size < (max(highs) - min(lows)) * 0.1:
                    patterns.append({"pattern": "RECTANGLE", "strength": 0.5, "direction": "NEUTRAL"})

        # --- Broadening Formation (Megaphone) ---
        if len(swing_highs) >= 3 and len(swing_lows) >= 3:
            recent_sh = swing_highs[-4:]
            recent_sl = swing_lows[-4:]
            if len(recent_sh) >= 2 and len(recent_sl) >= 2:
                x = list(range(len(recent_sh)))
                y_h = [highs[i] for i in recent_sh]
                y_l = [lows[i] for i in recent_sl]
                s_h, _ = self._linear_regression(x, y_h)
                s_l, _ = self._linear_regression(x, y_l)
                if s_h > 0.001 and s_l < -0.001:
                    patterns.append({"pattern": "BROADENING_FORMATION", "strength": 0.55, "direction": "VOLATILE"})

        # --- Diamond Pattern ---
        if len(swing_highs) >= 4 and len(swing_lows) >= 4:
            sh = swing_highs[-4:]
            sl = swing_lows[-4:]
            if len(sh) == 4 and len(sl) == 4:
                h_trend1 = highs[sh[1]] > highs[sh[0]] and highs[sh[2]] < highs[sh[1]]
                h_trend2 = highs[sh[3]] < highs[sh[2]]
                l_trend1 = lows[sl[1]] < lows[sl[0]] and lows[sl[2]] > lows[sl[1]]
                l_trend2 = lows[sl[3]] > lows[sl[2]]
                if h_trend1 and h_trend2 and l_trend1 and l_trend2:
                    patterns.append({"pattern": "DIAMOND_TOP", "strength": 0.7, "direction": "BEARISH"})

                h_trend3 = highs[sh[1]] < highs[sh[0]] and highs[sh[2]] > highs[sh[1]]
                h_trend4 = highs[sh[3]] > highs[sh[2]]
                l_trend3 = lows[sl[1]] > lows[sl[0]] and lows[sl[2]] < lows[sl[1]]
                l_trend4 = lows[sl[3]] < lows[sl[2]]
                if h_trend3 and h_trend4 and l_trend3 and l_trend4:
                    patterns.append({"pattern": "DIAMOND_BOTTOM", "strength": 0.7, "direction": "BULLISH"})

        # --- Bump and Run Reversal ---
        if len(candles) >= 30:
            first_10 = candles[:10]
            mid_10 = candles[10:20]
            last_10 = candles[-10:]
            first_slope = (first_10[-1]["close"] - first_10[0]["close"]) / 10
            mid_slope = (mid_10[-1]["close"] - mid_10[0]["close"]) / 10
            last_slope = (last_10[-1]["close"] - last_10[0]["close"]) / 10

            if first_slope > 0 and mid_slope > first_slope * 1.5 and last_slope < 0:
                patterns.append({"pattern": "BUMP_AND_RUN_TOP", "strength": 0.65, "direction": "BEARISH"})
            if first_slope < 0 and mid_slope < first_slope * 1.5 and last_slope > 0:
                patterns.append({"pattern": "BUMP_AND_RUN_BOTTOM", "strength": 0.65, "direction": "BULLISH"})

        # --- Island Reversal ---
        if len(candles) >= 10:
            for i in range(5, len(candles) - 5):
                c_prev = candles[i-1]
                c_curr = candles[i]
                c_next = candles[i+1]
                # Gap up then gap down, isolated candle(s)
                if c_curr["low"] > c_prev["high"] and c_next["high"] < c_curr["low"]:
                    if c_curr["close"] > c_curr["open"]:
                        patterns.append({"pattern": "ISLAND_REVERSAL_BULLISH", "strength": 0.7, "direction": "BULLISH", "index": i})
                    else:
                        patterns.append({"pattern": "ISLAND_REVERSAL_BEARISH", "strength": 0.7, "direction": "BEARISH", "index": i})

        # --- V-Bottom / V-Top ---
        if len(swing_lows) >= 2 and len(swing_highs) >= 1:
            for i in range(len(swing_lows) - 1):
                if lows[swing_lows[i+1]] > lows[swing_lows[i]] * 1.03:
                    # Sharp drop then sharp recovery
                    drop = (highs[swing_highs[i]] - lows[swing_lows[i]]) / highs[swing_highs[i]] if i < len(swing_highs) else 0
                    if drop > 0.05:
                        patterns.append({"pattern": "V_BOTTOM", "strength": 0.6, "direction": "BULLISH"})

        if len(swing_highs) >= 2 and len(swing_lows) >= 1:
            for i in range(len(swing_highs) - 1):
                if highs[swing_highs[i+1]] < highs[swing_highs[i]] * 0.97:
                    rise = (highs[swing_highs[i]] - lows[swing_lows[i]]) / lows[swing_lows[i]] if i < len(swing_lows) else 0
                    if rise > 0.05:
                        patterns.append({"pattern": "V_TOP", "strength": 0.6, "direction": "BEARISH"})

        # --- Adam & Eve Double Bottom ---
        if len(swing_lows) >= 2:
            for i in range(len(swing_lows) - 1):
                l1, l2 = lows[swing_lows[i]], lows[swing_lows[i+1]]
                if abs(l1 - l2) < (max(highs) - min(lows)) * 0.03:
                    # Adam = sharp V, Eve = rounded U
                    shape1 = abs(highs[swing_lows[i]] - min(lows[swing_lows[i]-2:swing_lows[i]+3])) if swing_lows[i] >= 2 else 0
                    shape2 = abs(highs[swing_lows[i+1]] - min(lows[swing_lows[i+1]-2:swing_lows[i+1]+3])) if swing_lows[i+1] >= 2 else 0
                    if shape1 > shape2 * 1.3:
                        patterns.append({"pattern": "ADAM_EVE_DOUBLE_BOTTOM", "strength": 0.7, "direction": "BULLISH"})
                    elif shape2 > shape1 * 1.3:
                        patterns.append({"pattern": "EVE_ADAM_DOUBLE_BOTTOM", "strength": 0.7, "direction": "BULLISH"})

        # --- Measured Move ---
        if len(candles) >= 30:
            first_10 = candles[:10]
            mid_10 = candles[10:20]
            last_10 = candles[-10:]
            first_move = abs(first_10[-1]["close"] - first_10[0]["close"])
            mid_retrace = abs(mid_10[-1]["close"] - mid_10[0]["close"])
            last_move = abs(last_10[-1]["close"] - last_10[0]["close"])
            if first_move > 0 and mid_retrace > 0 and abs(last_move - first_move) / first_move < 0.3:
                if first_10[-1]["close"] > first_10[0]["close"] and last_10[-1]["close"] > last_10[0]["close"]:
                    patterns.append({"pattern": "MEASURED_MOVE_UP", "strength": 0.6, "direction": "BULLISH"})
                elif first_10[-1]["close"] < first_10[0]["close"] and last_10[-1]["close"] < last_10[0]["close"]:
                    patterns.append({"pattern": "MEASURED_MOVE_DOWN", "strength": 0.6, "direction": "BEARISH"})

        # --- Saucer / Scallop ---
        if len(candles) >= 30:
            mid = len(candles) // 2
            left = candles[:mid]
            right = candles[mid:]
            left_trend = (left[-1]["close"] - left[0]["close"]) / left[0]["close"] if left[0]["close"] > 0 else 0
            right_trend = (right[-1]["close"] - right[0]["close"]) / right[0]["close"] if right[0]["close"] > 0 else 0

            if left_trend < -0.02 and right_trend > 0.02:
                patterns.append({"pattern": "SAUCER_BOTTOM", "strength": 0.55, "direction": "BULLISH"})
            if left_trend > 0.02 and right_trend < -0.02:
                patterns.append({"pattern": "SAUCER_TOP", "strength": 0.55, "direction": "BEARISH"})

        # --- Dome ---
        if len(candles) >= 30:
            mid = len(candles) // 2
            left = candles[:mid]
            right = candles[mid:]
            left_avg = sum(c["close"] for c in left) / len(left)
            right_avg = sum(c["close"] for c in right) / len(right)
            mid_avg = sum(c["close"] for c in candles[mid-5:mid+5]) / 10
            if mid_avg > left_avg * 1.02 and mid_avg > right_avg * 1.02:
                patterns.append({"pattern": "DOME", "strength": 0.5, "direction": "BEARISH"})

        # --- Right-Angled Broadening ---
        if len(swing_highs) >= 3 and len(swing_lows) >= 3:
            recent_sh = swing_highs[-4:]
            recent_sl = swing_lows[-4:]
            if len(recent_sh) >= 2 and len(recent_sl) >= 2:
                x = list(range(len(recent_sh)))
                y_h = [highs[i] for i in recent_sh]
                y_l = [lows[i] for i in recent_sl]
                s_h, _ = self._linear_regression(x, y_h)
                s_l, _ = self._linear_regression(x, y_l)
                if abs(s_h) < 0.001 and s_l < -0.001:
                    patterns.append({"pattern": "RIGHT_ANGLED_BROADENING_TOP", "strength": 0.55, "direction": "BEARISH"})
                if abs(s_h) < 0.001 and s_l > 0.001:
                    patterns.append({"pattern": "RIGHT_ANGLED_BROADENING_BOTTOM", "strength": 0.55, "direction": "BULLISH"})

        # --- Scallop ---
        if len(candles) >= 25:
            for i in range(10, len(candles) - 10):
                window = candles[i-10:i+10]
                mid = window[10]["close"]
                left_avg = sum(c["close"] for c in window[:10]) / 10
                right_avg = sum(c["close"] for c in window[10:]) / 10
                if left_avg < mid * 0.98 and right_avg > mid * 1.01:
                    patterns.append({"pattern": "SCALLOP", "strength": 0.5, "direction": "BULLISH"})

        score = min(1.0, len(patterns) * 0.15)

        return {
            "patterns": patterns,
            "score": round(score, 2),
            "explanation": f"{len(patterns)} chart patterns detected: {[p['pattern'] for p in patterns]}."
        }

    # ==================== LAYER 10: VOLUME ANALYTICS ====================
    def _layer10_volume(self, candles: List[Dict]) -> Dict[str, Any]:
        if len(candles) < 20:
            return {"spikes": [], "accumulation": False, "distribution": False, "participation": 0, "score": 0.5}

        volumes = [c["volume"] for c in candles]
        avg_vol = sum(volumes[-20:]) / 20
        last_vol = volumes[-1]

        # Volume spikes
        spikes = []
        for i in range(len(volumes) - 5, len(volumes)):
            if volumes[i] > avg_vol * 2:
                spikes.append({"index": i, "volume": volumes[i], "ratio": volumes[i] / avg_vol})

        # Accumulation vs Distribution (simplified: volume on up vs down candles)
        up_vol = sum(candles[i]["volume"] for i in range(len(candles)-20, len(candles)) if candles[i]["close"] > candles[i]["open"])
        down_vol = sum(candles[i]["volume"] for i in range(len(candles)-20, len(candles)) if candles[i]["close"] < candles[i]["open"])

        accumulation = up_vol > down_vol * 1.5
        distribution = down_vol > up_vol * 1.5

        participation = min(100, (last_vol / avg_vol) * 50) if avg_vol > 0 else 0

        score = 0.5
        if accumulation:
            score = 0.8
        elif distribution:
            score = 0.2
        if len(spikes) > 0:
            score += 0.1
        score = min(1.0, max(0.0, score))

        return {
            "spikes": spikes,
            "accumulation": accumulation,
            "distribution": distribution,
            "participation": round(participation, 2),
            "avg_volume": round(avg_vol, 2),
            "confirmation_engine": True,
            "score": round(score, 2),
            "explanation": f"Volume participation: {participation:.1f}%. {'Accumulation' if accumulation else 'Distribution' if distribution else 'Neutral'} detected. {len(spikes)} recent spikes. Volume confirmation engine active."
        }

    # ==================== LAYER 11: OPEN INTEREST ANALYTICS ====================
    def _layer11_open_interest(self, oi_data: List[Dict], candles: List[Dict]) -> Dict[str, Any]:
        if not oi_data or len(oi_data) < 5:
            return {"trend": "NEUTRAL", "liquidation_risk": 0, "score": 0.5}

        oi_values = [d["oi"] for d in oi_data]
        oi_change = (oi_values[-1] - oi_values[0]) / (oi_values[0] + 1e-9) * 100

        # Trend confirmation: OI rising with price = trend strengthening
        price_change = 0
        if candles and len(candles) >= len(oi_data):
            price_change = (candles[-1]["close"] - candles[-len(oi_data)]["close"]) / (candles[-len(oi_data)]["close"] + 1e-9) * 100

        if oi_change > 5 and price_change > 2:
            trend = "BULLISH_CONFIRMATION"
        elif oi_change > 5 and price_change < -2:
            trend = "BEARISH_DIVERGENCE"
        elif oi_change < -5 and price_change < -2:
            trend = "BEARISH_CONFIRMATION"
        elif oi_change < -5 and price_change > 2:
            trend = "BULLISH_DIVERGENCE"
        else:
            trend = "NEUTRAL"

        # Liquidation risk: high OI + high volatility
        liquidation_risk = 0
        if candles and len(candles) >= 20:
            atr = self._atr(candles, 14)
            if atr and atr[-1] > 0:
                volatility = atr[-1] / candles[-1]["close"] * 100
                liquidation_risk = min(100, volatility * (oi_values[-1] / max(oi_values)) * 10)

        score = 0.5
        if "CONFIRMATION" in trend:
            score = 0.8
        elif "DIVERGENCE" in trend:
            score = 0.3

        return {
            "trend": trend,
            "oi_change": round(oi_change, 2),
            "liquidation_risk": round(liquidation_risk, 2),
            "score": round(score, 2),
            "explanation": f"OI trend: {trend}. OI change: {oi_change:.2f}%. Liquidation risk: {liquidation_risk:.1f}%."
        }

    # ==================== LAYER 12: FUNDING RATE ANALYTICS ====================
    def _layer12_funding(self, funding_data: List[Dict]) -> Dict[str, Any]:
        if not funding_data or len(funding_data) < 5:
            return {"sentiment": "NEUTRAL", "extreme": False, "contrarian_signal": False, "score": 0.5}

        rates = [d["rate"] for d in funding_data]
        avg_rate = sum(rates) / len(rates)
        last_rate = rates[-1]
        max_rate = max(rates)
        min_rate = min(rates)

        # Extreme detection
        extreme_threshold = 0.001  # 0.1%
        is_extreme = abs(last_rate) > extreme_threshold

        # Sentiment imbalance detection
        imbalance_ratio = abs(last_rate) / (abs(avg_rate) + 1e-9) if avg_rate != 0 else 0
        sentiment_imbalance = imbalance_ratio > 2.0  # Current rate is 2x+ the average

        # Sentiment
        if last_rate > 0.0005:
            sentiment = "LONG_HEAVY"
        elif last_rate < -0.0005:
            sentiment = "SHORT_HEAVY"
        else:
            sentiment = "BALANCED"

        # Contrarian signal: extreme funding often leads to reversal
        contrarian = is_extreme and abs(last_rate) == max(abs(r) for r in rates)

        score = 0.5
        if contrarian:
            score = 0.3 if last_rate > 0 else 0.7  # Contrarian to long-heavy = bullish, to short-heavy = bearish
        elif sentiment == "BALANCED":
            score = 0.5
        elif sentiment == "LONG_HEAVY":
            score = 0.3
        else:
            score = 0.7

        return {
            "sentiment": sentiment,
            "current_rate": round(last_rate * 100, 4),
            "avg_rate": round(avg_rate * 100, 4),
            "extreme": is_extreme,
            "contrarian_signal": contrarian,
            "sentiment_imbalance": sentiment_imbalance,
            "imbalance_ratio": round(imbalance_ratio, 2),
            "score": round(score, 2),
            "explanation": f"Funding: {sentiment}. Current: {last_rate*100:.4f}%. {'Sentiment imbalance detected!' if sentiment_imbalance else 'Balanced sentiment.'} {'Contrarian opportunity.' if contrarian else ''}"
        }

    # ==================== LAYER 13: LONG/SHORT RATIO ====================
    def _layer13_long_short(self, ls_data: List[Dict]) -> Dict[str, Any]:
        if not ls_data or len(ls_data) < 3:
            return {"ratio": 1.0, "crowd_position": "NEUTRAL", "sentiment_pressure": 0, "score": 0.5}

        ratios = [d["long_short_ratio"] for d in ls_data]
        avg_ratio = sum(ratios) / len(ratios)
        last_ratio = ratios[-1]

        if last_ratio > 2.0:
            crowd = "EXTREME_LONG"
        elif last_ratio > 1.5:
            crowd = "LONG_HEAVY"
        elif last_ratio < 0.5:
            crowd = "EXTREME_SHORT"
        elif last_ratio < 0.75:
            crowd = "SHORT_HEAVY"
        else:
            crowd = "BALANCED"

        # Sentiment pressure: deviation from 1.0
        pressure = abs(last_ratio - 1.0) / 1.0 * 100

        # Contrarian: crowd is usually wrong at extremes
        score = 0.5
        if crowd in ["EXTREME_LONG", "LONG_HEAVY"]:
            score = 0.3  # Bearish bias (crowd long = potential down)
        elif crowd in ["EXTREME_SHORT", "SHORT_HEAVY"]:
            score = 0.7  # Bullish bias (crowd short = potential up)

        return {
            "ratio": round(last_ratio, 2),
            "avg_ratio": round(avg_ratio, 2),
            "crowd_position": crowd,
            "sentiment_pressure": round(pressure, 2),
            "score": round(score, 2),
            "explanation": f"L/S ratio: {last_ratio:.2f}. Crowd is {crowd.lower()}. Pressure: {pressure:.1f}%."
        }

    # ==================== LAYER 14: SESSION ANALYSIS ====================
    def _layer14_session(self, candles: List[Dict]) -> Dict[str, Any]:
        if len(candles) < 24:
            return {"current_session": "UNKNOWN", "volatility_by_session": {}, "liquidity_by_session": {}, "score": 0.5}

        # Determine current session based on last candle time (UTC)
        last_time = candles[-1]["time"]
        dt = datetime.utcfromtimestamp(last_time / 1000)
        hour = dt.hour

        if 0 <= hour < 8:
            current = "ASIAN"
        elif 8 <= hour < 16:
            current = "LONDON"
        else:
            current = "NEW_YORK"

        # Analyze by session (simplified: group candles by UTC hour)
        session_volatility = {"ASIAN": [], "LONDON": [], "NEW_YORK": []}
        session_volume = {"ASIAN": [], "LONDON": [], "NEW_YORK": []}

        for c in candles:
            h = datetime.utcfromtimestamp(c["time"] / 1000).hour
            vol = c["high"] - c["low"]
            if 0 <= h < 8:
                session_volatility["ASIAN"].append(vol)
                session_volume["ASIAN"].append(c["volume"])
            elif 8 <= h < 16:
                session_volatility["LONDON"].append(vol)
                session_volume["LONDON"].append(c["volume"])
            else:
                session_volatility["NEW_YORK"].append(vol)
                session_volume["NEW_YORK"].append(c["volume"])

        vol_by_session = {k: round(sum(v) / len(v), 2) if v else 0 for k, v in session_volatility.items()}
        liq_by_session = {k: round(sum(v) / len(v), 2) if v else 0 for k, v in session_volume.items()}

        # Score: London and NY have better liquidity
        score = 0.5
        if current in ["LONDON", "NEW_YORK"]:
            score = 0.8
        elif current == "ASIAN":
            score = 0.6

        return {
            "current_session": current,
            "volatility_by_session": vol_by_session,
            "liquidity_by_session": liq_by_session,
            "score": round(score, 2),
            "explanation": f"Current session: {current}. London vol: {vol_by_session.get('LONDON', 0):.2f}, NY vol: {vol_by_session.get('NEW_YORK', 0):.2f}."
        }

    # ==================== LAYER 15: MULTI-TIMEFRAME ALIGNMENT ====================
    def _layer15_multitimeframe(self, tf_data: Dict[str, List[Dict]], primary_tf: str) -> Dict[str, Any]:
        if not tf_data:
            return {"alignment": "NEUTRAL", "agreement_score": 0, "conflicts": [], "score": 0.5}

        directions = {}
        for tf, candles in tf_data.items():
            if len(candles) < 20:
                continue
            closes = [c["close"] for c in candles]
            ema20 = self._ema(closes, 20)
            ema50 = self._ema(closes, 50)
            if ema20[-1] > ema50[-1]:
                directions[tf] = "BULLISH"
            elif ema20[-1] < ema50[-1]:
                directions[tf] = "BEARISH"
            else:
                directions[tf] = "NEUTRAL"

        # Add primary timeframe
        directions[primary_tf] = directions.get(primary_tf, "NEUTRAL")

        if not directions:
            return {"alignment": "NEUTRAL", "agreement_score": 0, "conflicts": [], "score": 0.5}

        bullish_count = sum(1 for d in directions.values() if d == "BULLISH")
        bearish_count = sum(1 for d in directions.values() if d == "BEARISH")
        total = len(directions)

        if bullish_count > bearish_count and bullish_count >= total * 0.6:
            alignment = "BULLISH_ALIGNED"
        elif bearish_count > bullish_count and bearish_count >= total * 0.6:
            alignment = "BEARISH_ALIGNED"
        elif bullish_count == bearish_count:
            alignment = "CONFLICTED"
        else:
            alignment = "MIXED"

        agreement_score = max(bullish_count, bearish_count) / total * 100 if total > 0 else 0

        conflicts = []
        primary_dir = directions.get(primary_tf, "NEUTRAL")
        for tf, dir in directions.items():
            if tf != primary_tf and dir != primary_dir and dir != "NEUTRAL" and primary_dir != "NEUTRAL":
                conflicts.append({"timeframe": tf, "direction": dir, "primary": primary_dir})

        score = agreement_score / 100

        return {
            "alignment": alignment,
            "directions": directions,
            "agreement_score": round(agreement_score, 2),
            "conflicts": conflicts,
            "score": round(score, 2),
            "explanation": f"Multi-TF alignment: {alignment}. Agreement: {agreement_score:.1f}%. {len(conflicts)} conflicts."
        }

    # ==================== LAYER 16: MARKET REGIME ENGINE ====================
    def _layer16_regime(self, candles: List[Dict]) -> Dict[str, Any]:
        if len(candles) < 50:
            return {"regime": "UNKNOWN", "expansion": False, "contraction": False, "score": 0.5}

        closes = [c["close"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]

        atr = self._atr(candles, 14)
        if not atr or len(atr) < 20:
            return {"regime": "UNKNOWN", "score": 0.5}

        recent_atr = atr[-20:]
        atr_trend = (recent_atr[-1] - recent_atr[0]) / (recent_atr[0] + 1e-9)

        # Volatility regime
        if atr_trend > 0.3:
            expansion = True
            contraction = False
        elif atr_trend < -0.3:
            expansion = False
            contraction = True
        else:
            expansion = False
            contraction = False

        # Trending vs Ranging via ADX-like
        ema20 = self._ema(closes, 20)
        ema50 = self._ema(closes, 50)

        if abs(ema20[-1] - ema50[-1]) / (closes[-1] + 1e-9) > 0.02 and not contraction:
            regime = "TRENDING"
        elif contraction and abs(ema20[-1] - ema50[-1]) / (closes[-1] + 1e-9) < 0.01:
            regime = "RANGING"
        elif expansion:
            regime = "VOLATILE"
        else:
            regime = "TRANSITIONING"

        score = 0.5
        if regime == "TRENDING":
            score = 0.8
        elif regime == "RANGING":
            score = 0.4
        elif regime == "VOLATILE":
            score = 0.3

        return {
            "regime": regime,
            "expansion": expansion,
            "contraction": contraction,
            "score": round(score, 2),
            "explanation": f"Market regime: {regime}. {'Expansion' if expansion else 'Contraction' if contraction else 'Stable'} phase."
        }

    # ==================== LAYER 17: VOLATILITY ENGINE ====================
    def _layer17_volatility(self, candles: List[Dict]) -> Dict[str, Any]:
        if len(candles) < 20:
            return {"atr": 0, "atr_percent": 0, "ranking": "LOW", "expansion": False, "risk_class": "LOW", "score": 0.5}

        closes = [c["close"] for c in candles]
        atr = self._atr(candles, 14)
        current_atr = atr[-1] if atr else 0
        atr_percent = (current_atr / closes[-1] * 100) if closes[-1] > 0 else 0

        # Volatility ranking
        if atr_percent > 5:
            ranking = "EXTREME"
        elif atr_percent > 3:
            ranking = "HIGH"
        elif atr_percent > 1.5:
            ranking = "MODERATE"
        else:
            ranking = "LOW"

        # Expansion/contraction
        if len(atr) >= 20:
            atr_change = (atr[-1] - atr[-20]) / (atr[-20] + 1e-9)
            expansion = atr_change > 0.2
        else:
            expansion = False

        # Risk classification
        if ranking in ["EXTREME", "HIGH"]:
            risk_class = "HIGH"
        elif ranking == "MODERATE":
            risk_class = "MODERATE"
        else:
            risk_class = "LOW"

        score = 0.5
        if ranking == "MODERATE":
            score = 0.8  # Sweet spot for trading
        elif ranking == "LOW":
            score = 0.6
        elif ranking == "HIGH":
            score = 0.4
        else:
            score = 0.2

        return {
            "atr": round(current_atr, 4),
            "atr_percent": round(atr_percent, 2),
            "ranking": ranking,
            "expansion": expansion,
            "risk_class": risk_class,
            "score": round(score, 2),
            "explanation": f"ATR: {current_atr:.4f} ({atr_percent:.2f}%). Ranking: {ranking}. Risk: {risk_class}."
        }

    # ==================== LAYER 18: RMT ANALYTICS ENGINE ====================
    def _layer18_rmt(self, candles: List[Dict]) -> Dict[str, Any]:
        if len(candles) < 60:
            return {"eigenvalues": [], "noise_threshold": 0, "signals": 0, "complexity": 0, "regime_shift": False, "score": 0.5}

        closes = [c["close"] for c in candles]

        # Calculate log returns
        returns = self._returns(closes)
        if len(returns) < 30:
            return {"score": 0.5}

        # Build covariance matrix from rolling windows
        window_size = min(30, len(returns) // 2)
        n_windows = len(returns) - window_size + 1

        if n_windows < 5:
            return {"score": 0.5}

        # Create matrix of return windows
        matrix = []
        for i in range(n_windows):
            window = returns[i:i+window_size]
            matrix.append(window)

        matrix = np.array(matrix)
        if matrix.shape[0] < matrix.shape[1]:
            matrix = matrix.T

        # Covariance matrix
        try:
            cov_matrix = np.cov(matrix)
            if cov_matrix.ndim < 2:
                return {"score": 0.5}

            eigenvalues = np.linalg.eigvalsh(cov_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]

            # Marchenko-Pastur threshold for noise filtering
            # Eigenvalues above threshold = signal (meaningful market structure)
            # Eigenvalues below threshold = noise (random market fluctuations)
            N = len(eigenvalues)
            T = matrix.shape[0]
            if T > N and N > 0:
                q = T / N
                sigma2 = np.mean(eigenvalues)
                lambda_max = sigma2 * (1 + 1/q + 2 * np.sqrt(1/q))

                # Noise filtering: separate signal from noise
                signals = int(np.sum(eigenvalues > lambda_max))
                noise = int(np.sum(eigenvalues <= lambda_max))
                noise_filtered = True
            else:
                lambda_max = np.median(eigenvalues)
                signals = int(np.sum(eigenvalues > lambda_max * 1.5))
                noise = len(eigenvalues) - signals
                noise_filtered = True

            # Signal extraction: top eigenvalues represent dominant market factors
            signal_eigenvalues = eigenvalues[:signals] if signals > 0 else []
            noise_eigenvalues = eigenvalues[signals:] if signals < len(eigenvalues) else []
            signal_power = np.sum(signal_eigenvalues) if len(signal_eigenvalues) > 0 else 0
            noise_power = np.sum(noise_eigenvalues) if len(noise_eigenvalues) > 0 else 0
            snr = signal_power / (noise_power + 1e-9)  # Signal-to-noise ratio

            # Complexity score: ratio of signal to total
            complexity = signals / len(eigenvalues) if len(eigenvalues) > 0 else 0

            # Regime shift: sudden change in top eigenvalue ratio
            regime_shift = False
            if len(eigenvalues) >= 4:
                top_ratio = eigenvalues[0] / (eigenvalues[1] + 1e-9)
                regime_shift = top_ratio > 3.0  # Dominant factor emergence

            score = min(1.0, complexity * 2 + (0.2 if regime_shift else 0))

            return {
                "eigenvalues": [round(float(e), 4) for e in eigenvalues[:5]],
                "noise_threshold": round(float(lambda_max), 4),
                "signals": signals,
                "noise_components": noise,
                "signal_power": round(float(signal_power), 4),
                "noise_power": round(float(noise_power), 4),
                "snr": round(float(snr), 4),
                "noise_filtered": noise_filtered,
                "complexity": round(float(complexity), 4),
                "regime_shift": bool(regime_shift),
                "score": round(float(score), 2),
                "explanation": f"RMT: {signals} signals extracted from {noise} noise components. SNR: {snr:.2f}. Noise filtering applied. Complexity: {complexity:.2%}. {'Regime shift detected.' if regime_shift else 'Stable regime.'}"
            }
        except Exception as e:
            return {"score": 0.5, "explanation": f"RMT calculation error: {str(e)}"}

    # ==================== LAYER 19: CONFLUENCE SCORE ====================
    def _layer19_confluence(self, layers: Dict[str, Any]) -> Dict[str, Any]:
        weighted_sum = 0
        total_weight = 0

        for layer_name, weight in self.layer_weights.items():
            if layer_name in layers and "score" in layers[layer_name]:
                weighted_sum += layers[layer_name]["score"] * weight
                total_weight += weight

        if total_weight == 0:
            confluence = 0
        else:
            confluence = (weighted_sum / total_weight) * 10

        # Directional bias from layers
        bullish_signals = 0
        bearish_signals = 0

        for layer_name, layer_data in layers.items():
            if "direction" in layer_data:
                if "BULLISH" in str(layer_data["direction"]):
                    bullish_signals += 1
                elif "BEARISH" in str(layer_data["direction"]):
                    bearish_signals += 1
            if "trend" in layer_data and isinstance(layer_data["trend"], str):
                if "UP" in layer_data["trend"] or "BULLISH" in layer_data["trend"]:
                    bullish_signals += 1
                elif "DOWN" in layer_data["trend"] or "BEARISH" in layer_data["trend"]:
                    bearish_signals += 1

        if bullish_signals > bearish_signals * 1.5:
            direction = "BULLISH"
        elif bearish_signals > bullish_signals * 1.5:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        return {
            "confluence_score": round(confluence, 2),
            "direction": direction,
            "bullish_signals": bullish_signals,
            "bearish_signals": bearish_signals,
            "explanation": f"Confluence: {confluence:.1f}/10. Direction: {direction}. Bullish: {bullish_signals}, Bearish: {bearish_signals}."
        }

    # ==================== LAYER 20: SETUP QUALITY ENGINE ====================
    def _layer20_setup_quality(self, confluence: float, confidence: float, layers: Dict) -> Dict[str, Any]:
        # Grade based on confluence AND alignment
        quality = "POOR"
        mtf_alignment = layers.get("multi_timeframe", {}).get("agreement_score", 0)

        if confluence >= 8.0 and confidence >= 85 and mtf_alignment >= 75:
            quality = "ELITE"
        elif confluence >= 7.0 and confidence >= 75 and mtf_alignment >= 60:
            quality = "STRONG"
        elif confluence >= 6.0 and confidence >= 65 and mtf_alignment >= 50:
            quality = "GOOD"
        elif confluence >= 5.0 and confidence >= 50:
            quality = "MODERATE"
        elif confluence >= 4.0:
            quality = "WEAK"

        # Check critical layers
        critical_ok = True
        if layers.get("liquidity", {}).get("score", 0) < 0.2:
            critical_ok = False
        if layers.get("market_structure", {}).get("score", 0) < 0.3:
            critical_ok = False

        if not critical_ok and quality in ["GOOD", "STRONG", "ELITE"]:
            quality = "MODERATE"

        return {
            "quality": quality,
            "criteria_met": {
                "confluence": confluence >= 6.0,
                "confidence": confidence >= 65,
                "liquidity": layers.get("liquidity", {}).get("score", 0) >= 0.2,
                "structure": layers.get("market_structure", {}).get("score", 0) >= 0.3,
                "rmt": layers.get("rmt", {}).get("score", 0) >= 0.3
            },
            "explanation": f"Setup quality: {quality} based on confluence ({confluence:.1f}) and multi-timeframe alignment ({mtf_alignment:.0f}%). {'All criteria met.' if all([confluence >= 6.0, confidence >= 65]) else 'Missing criteria.'}"
        }

    # ==================== LAYER 21: CONFIDENCE METER ====================
    def _layer21_confidence(self, layers: Dict, confluence: float) -> Dict[str, Any]:
        base = confluence * 10

        # Adjust based on alignment
        mtf = layers.get("multi_timeframe", {})
        agreement = mtf.get("agreement_score", 50)
        base *= (0.5 + agreement / 200)

        # Adjust based on RMT
        rmt = layers.get("rmt", {})
        if rmt.get("regime_shift", False):
            base *= 0.9  # Slightly reduce during regime shifts

        # Adjust based on volatility
        vol = layers.get("volatility", {})
        if vol.get("ranking", "LOW") == "EXTREME":
            base *= 0.7

        confidence = min(100, max(0, base))

        return {
            "confidence": round(confidence, 2),
            "gauge": "HIGH" if confidence >= 75 else "MODERATE" if confidence >= 50 else "LOW",
            "explanation": f"Confidence: {confidence:.1f}/100. {'High conviction.' if confidence >= 75 else 'Moderate conviction.' if confidence >= 50 else 'Low conviction, exercise caution.'}"
        }

    # ==================== LAYER 22: BACKTESTING ENGINE ====================
        async def backtest(self, candles: List[Dict], symbol: str, timeframe: str) -> Dict[str, Any]: 
            if len(candles) < 50:
            return {"error": "Insufficient data"}

            trades = []
            position = None
            entry_price = 0
            stop_loss = 0
            take_profit = 0

        for i in range(50, len(candles) - 1):
            window = candles[:i+1]

            # Simple strategy: buy on bullish structure + liquidity sweep, sell on bearish
            l1 = self._layer1_trend_range(window)
            l2 = self._layer2_market_structure(window)
            l3 = self._layer3_liquidity(window)

            signal = "NEUTRAL"
            if l1["direction"] == "BULLISH" and l2["structure"] in ["BULLISH", "BULLISH_TRANSITION"] and l3["score"] > 0.3:
                signal = "BUY"
            elif l1["direction"] == "BEARISH" and l2["structure"] in ["BEARISH", "BEARISH_TRANSITION"] and l3["score"] > 0.3:
                signal = "SELL"

            current_price = candles[i]["close"]

            if position is None:
                if signal == "BUY":
                    position = "LONG"
                    entry_price = current_price
                    stop_loss = entry_price - (entry_price * 0.02)  # 2% stop
                    take_profit = entry_price + (entry_price * 0.04)  # 2:1 RR
                elif signal == "SELL":
                    position = "SHORT"
                    entry_price = current_price
                    stop_loss = entry_price + (entry_price * 0.02)
                    take_profit = entry_price - (entry_price * 0.04)
            else:
                # Check exit
                if position == "LONG":
                    if current_price <= stop_loss:
                        trades.append({"type": "LONG", "entry": entry_price, "exit": current_price, "pnl": -0.02, "result": "LOSS"})
                        position = None
                    elif current_price >= take_profit:
                        trades.append({"type": "LONG", "entry": entry_price, "exit": current_price, "pnl": 0.04, "result": "WIN"})
                        position = None
                elif position == "SHORT":
                    if current_price >= stop_loss:
                        trades.append({"type": "SHORT", "entry": entry_price, "exit": current_price, "pnl": -0.02, "result": "LOSS"})
                        position = None
                    elif current_price <= take_profit:
                        trades.append({"type": "SHORT", "entry": entry_price, "exit": current_price, "pnl": 0.04, "result": "WIN"})
                        position = None

        if not trades:
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "strategy": "RMT_Structure_Liquidity",
                "start_date": datetime.utcfromtimestamp(candles[0]["time"]/1000).isoformat(),
                "end_date": datetime.utcfromtimestamp(candles[-1]["time"]/1000).isoformat(),
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "avg_rr": 0,
                "performance_metrics": {},
                "explanation": "No trades generated in backtest period."
            }

        wins = [t for t in trades if t["result"] == "WIN"]
        losses = [t for t in trades if t["result"] == "LOSS"]
        win_rate = len(wins) / len(trades) * 100 if trades else 0
        total_profit = sum(t["pnl"] for t in wins)
        total_loss = abs(sum(t["pnl"] for t in losses))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        avg_rr = 2.0  # Fixed 1:2 RR

        # Performance metrics
        max_drawdown = 0
        peak = 0
        running_pnl = 0
        for t in trades:
            running_pnl += t["pnl"]
            if running_pnl > peak:
                peak = running_pnl
            drawdown = peak - running_pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        sharpe = (total_profit - total_loss) / (self._std([t["pnl"] for t in trades]) * len(trades)**0.5 + 1e-9) if trades else 0

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy": "RMT_Structure_Liquidity",
            "start_date": datetime.utcfromtimestamp(candles[0]["time"]/1000).isoformat(),
            "end_date": datetime.utcfromtimestamp(candles[-1]["time"]/1000).isoformat(),
            "total_trades": len(trades),
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "avg_rr": avg_rr,
            "max_drawdown": round(max_drawdown * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "performance_metrics": {
                "total_return": round(running_pnl * 100, 2),
                "avg_trade": round(running_pnl / len(trades) * 100, 2) if trades else 0,
                "max_drawdown_pct": round(max_drawdown * 100, 2),
                "sharpe": round(sharpe, 2)
            },
            "trades": trades[-20:],
            "explanation": f"Backtest performance: {len(trades)} trades, {win_rate:.1f}% win rate, PF: {profit_factor:.2f}, Sharpe: {sharpe:.2f}. Performance metrics included."
        }

    def _layer23_trade_plan(self, layers: Dict, symbol: str, current_price: float) -> Dict[str, Any]:
        direction = layers.get("confluence", {}).get("direction", "NEUTRAL")
        quality = layers.get("setup_quality", {}).get("quality", "POOR")

        if quality in ["POOR", "WEAK"] or direction == "NEUTRAL":
            return {
                "valid": False,
                "reason": f"Setup quality {quality} or direction {direction} not sufficient.",
                "explanation": "No valid trade plan. Criteria not met."
            }

        atr = layers.get("volatility", {}).get("atr", current_price * 0.01)

        if direction == "BULLISH":
            entry = current_price
            stop_loss = current_price - (atr * 1.5)
            take_profit_1 = current_price + (atr * 1.5)
            take_profit_2 = current_price + (atr * 3.0)
            invalidation = stop_loss - (atr * 0.5)
        else:
            entry = current_price
            stop_loss = current_price + (atr * 1.5)
            take_profit_1 = current_price - (atr * 1.5)
            take_profit_2 = current_price - (atr * 3.0)
            invalidation = stop_loss + (atr * 0.5)

        risk = abs(entry - stop_loss)
        reward = abs(take_profit_1 - entry)
        rr = reward / risk if risk > 0 else 0

        # Position size: 1% risk per trade
        account_size = 10000  # Default assumption
        risk_amount = account_size * 0.01
        position_size = risk_amount / risk if risk > 0 else 0

        return {
            "valid": True,
            "direction": direction,
            "entry": round(entry, 4),
            "stop_loss": round(stop_loss, 4),
            "take_profit_1": round(take_profit_1, 4),
            "take_profit_2": round(take_profit_2, 4),
            "invalidation": round(invalidation, 4),
            "risk_reward": round(rr, 2),
            "position_size": round(position_size, 4),
            "risk_percent": 1.0,
            "explanation": f"Trade plan: {direction} entry at {entry:.4f}, SL {stop_loss:.4f}, TP1 {take_profit_1:.4f}, RR {rr:.2f}."
        }

    # ==================== LAYER 24: AI SUMMARY ====================
    def _layer24_ai_summary(self, layers: Dict, symbol: str, timeframe: str) -> Dict[str, Any]:
        direction = layers.get("confluence", {}).get("direction", "NEUTRAL")
        confluence = layers.get("confluence", {}).get("confluence_score", 0)
        confidence = layers.get("confidence", {}).get("confidence", 0)
        quality = layers.get("setup_quality", {}).get("quality", "POOR")
        regime = layers.get("market_regime", {}).get("regime", "UNKNOWN")
        trend = layers.get("trend_range", {}).get("trend", "NEUTRAL")

        summary_parts = [f"Analysis for {symbol} on {timeframe} timeframe — Human-Readable Summary of All Findings:"]

        summary_parts.append(f"The market is currently in a {trend.lower()} regime with {regime.lower()} characteristics.")

        if direction == "BULLISH":
            summary_parts.append(f"Directional bias is BULLISH with {confluence:.1f}/10 confluence and {confidence:.0f}% confidence.")
        elif direction == "BEARISH":
            summary_parts.append(f"Directional bias is BEARISH with {confluence:.1f}/10 confluence and {confidence:.0f}% confidence.")
        else:
            summary_parts.append(f"Directional bias is NEUTRAL. Confluence {confluence:.1f}/10, confidence {confidence:.0f}%.")

        # Key observations
        observations = []
        if layers.get("liquidity", {}).get("sweeps", []):
            observations.append("liquidity sweeps detected")
        if layers.get("order_blocks", {}).get("fresh_blocks", []):
            observations.append("fresh order blocks present")
        if layers.get("fvg", {}).get("unfilled_fvgs", []):
            observations.append("unfilled fair value gaps")
        if layers.get("candlestick", {}).get("recent_patterns", []):
            observations.append("recent candlestick patterns")
        if layers.get("funding", {}).get("contrarian_signal", False):
            observations.append("funding rate contrarian signal")
        if layers.get("rmt", {}).get("regime_shift", False):
            observations.append("RMT regime shift detected")

        if observations:
            summary_parts.append(f"Key findings from the analysis include: {', '.join(observations)}.")

        summary_parts.append(f"Setup quality: {quality}.")

        if quality in ["ELITE", "STRONG", "GOOD"]:
            summary_parts.append("Setup meets minimum criteria for consideration.")
        else:
            summary_parts.append("Setup does NOT meet minimum criteria. Exercise caution or wait for better confluence.")

        summary_parts.append("This is a human-readable summary for educational purposes. This platform is a decision-support tool only — not financial advice. Always manage your risk carefully.")

        return {
            "summary": " ".join(summary_parts),
            "direction": direction,
            "confluence": confluence,
            "confidence": confidence,
            "quality": quality,
            "explanation": "AI-generated summary based on all 24 analysis layers."
        }

    # ==================== MAIN ANALYZE METHOD ====================
    async def analyze(self, symbol: str, timeframe: str, candles: List[Dict],
                      multi_timeframe_data: Dict[str, List[Dict]],
                      funding_data: List[Dict], oi_data: List[Dict],
                      ls_ratio_data: List[Dict], stats_24h: Dict) -> Dict[str, Any]:

        # Run all layers
        layers = {}
        layers["trend_range"] = self._layer1_trend_range(candles)
        layers["market_structure"] = self._layer2_market_structure(candles)
        layers["liquidity"] = self._layer3_liquidity(candles)
        layers["order_blocks"] = self._layer4_order_blocks(candles)
        layers["fvg"] = self._layer5_fvg(candles, multi_timeframe_data)
        layers["premium_discount"] = self._layer6_premium_discount(candles)
        layers["price_action"] = self._layer7_price_action(candles)
        layers["candlestick"] = self._layer8_candlestick(candles)
        layers["chart_pattern"] = self._layer9_chart_patterns(candles)
        layers["volume"] = self._layer10_volume(candles)
        layers["open_interest"] = self._layer11_open_interest(oi_data, candles)
        layers["funding"] = self._layer12_funding(funding_data)
        layers["long_short"] = self._layer13_long_short(ls_ratio_data)
        layers["session"] = self._layer14_session(candles)
        layers["multi_timeframe"] = self._layer15_multitimeframe(multi_timeframe_data, timeframe)
        layers["market_regime"] = self._layer16_regime(candles)
        layers["volatility"] = self._layer17_volatility(candles)
        layers["rmt"] = self._layer18_rmt(candles)

        # Composite layers
        layers["confluence"] = self._layer19_confluence(layers)
        confluence_score = layers["confluence"]["confluence_score"]

        layers["confidence"] = self._layer21_confidence(layers, confluence_score)
        confidence_score = layers["confidence"]["confidence"]

        layers["setup_quality"] = self._layer20_setup_quality(confluence_score, confidence_score, layers)

        current_price = candles[-1]["close"] if candles else 0
        layers["trade_plan"] = self._layer23_trade_plan(layers, symbol, current_price)

        layers["ai_summary"] = self._layer24_ai_summary(layers, symbol, timeframe)

        # Build final result
        result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": datetime.utcnow().isoformat(),
            "current_price": round(current_price, 4),
            "price_change_24h": round(stats_24h.get("price_change_percent", 0), 2),
            "volume_24h": round(stats_24h.get("volume", 0), 2),
            "direction": layers["confluence"]["direction"],
            "confluence_score": confluence_score,
            "confidence_score": confidence_score,
            "setup_quality": layers["setup_quality"]["quality"],
            "layers": layers,
            "explanations": {k: v.get("explanation", "") for k, v in layers.items()}
        }

        return result
