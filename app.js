// ===== RMT Crypto Trading Analysis Platform - Frontend =====
const API_BASE = '';

let currentResult = null;
let watchlist = [];

// ===== INIT =====
document.addEventListener('DOMContentLoaded', () => {
    loadDisclaimer();
    loadCriteria();
    loadWatchlist();
    loadHistory();
    loadSymbols();
});

// ===== API HELPERS =====
async function apiGet(endpoint) {
    const res = await fetch(API_BASE + endpoint);
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}

async function apiPost(endpoint) {
    const res = await fetch(API_BASE + endpoint, { method: 'POST' });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}

async function apiDelete(endpoint) {
    const res = await fetch(API_BASE + endpoint, { method: 'DELETE' });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
}

// ===== DISCLAIMER =====
async function loadDisclaimer() {
    try {
        const data = await apiGet('/api/disclaimer');
        document.getElementById('disclaimer-text').textContent = data.disclaimer;
        // Also add a note about this being a decision-support tool, not an auto-trading bot
        const note = document.createElement('div');
        note.style.cssText = 'margin-top:8px;font-size:0.8rem;color:var(--danger);font-weight:600;';
        note.textContent = '⚠️ This is a decision-support system, NOT an auto-trading bot.';
        document.getElementById('disclaimer-banner').appendChild(note);
    } catch (e) {
        document.getElementById('disclaimer-text').textContent = 'Educational purposes only. Not financial advice.';
    }
}

function toggleDisclaimer() {
    const banner = document.getElementById('disclaimer-banner');
    banner.classList.toggle('hidden');
}

// ===== CRITERIA =====
async function loadCriteria() {
    try {
        const data = await apiGet('/api/disclaimer');
        const grid = document.getElementById('criteria-grid');
        grid.innerHTML = '';
        for (const [key, value] of Object.entries(data.criteria)) {
            const div = document.createElement('div');
            div.className = 'criteria-item';
            div.innerHTML = `<span class="check">✓</span> <strong>${key.replace(/_/g, ' ').toUpperCase()}:</strong> ${value}`;
            grid.appendChild(div);
        }
    } catch (e) {
        console.error('Criteria load error:', e);
    }
}

// ===== SYMBOLS =====
async function loadSymbols() {
    try {
        const data = await apiGet('/api/symbols');
        const select = document.getElementById('symbol-select');
        const current = select.value;
        select.innerHTML = '';
        data.symbols.slice(0, 50).forEach(sym => {
            const opt = document.createElement('option');
            opt.value = sym;
            opt.textContent = sym;
            if (sym === current) opt.selected = true;
            select.appendChild(opt);
        });
    } catch (e) {
        console.error('Symbols load error:', e);
    }
}

// ===== WATCHLIST =====
async function loadWatchlist() {
    try {
        const data = await apiGet('/api/watchlist');
        watchlist = data.watchlist;
        renderWatchlist();
    } catch (e) {
        watchlist = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","ADAUSDT","DOGEUSDT","LINKUSDT","AVAXUSDT","MATICUSDT"];
        renderWatchlist();
    }
}

function renderWatchlist() {
    const container = document.getElementById('watchlist-tags');
    container.innerHTML = '';
    const currentSymbol = document.getElementById('symbol-select').value;
    watchlist.forEach(sym => {
        const tag = document.createElement('div');
        tag.className = 'watchlist-tag' + (sym === currentSymbol ? ' active' : '');
        tag.innerHTML = `<span onclick="selectSymbol('${sym}')">${sym}</span><span class="remove" onclick="removeWatchlist('${sym}', event)">×</span>`;
        container.appendChild(tag);
    });
}

function selectSymbol(sym) {
    document.getElementById('symbol-select').value = sym;
    renderWatchlist();
}

async function addToWatchlist() {
    const input = document.getElementById('new-symbol');
    const sym = input.value.trim().toUpperCase();
    if (!sym) return;
    try {
        await apiPost(`/api/watchlist/${sym}`);
        input.value = '';
        await loadWatchlist();
    } catch (e) {
        alert('Failed to add: ' + e.message);
    }
}

async function removeWatchlist(sym, event) {
    event.stopPropagation();
    try {
        await apiDelete(`/api/watchlist/${sym}`);
        await loadWatchlist();
    } catch (e) {
        console.error('Remove error:', e);
    }
}

// ===== ANALYSIS =====
async function runAnalysis() {
    const symbol = document.getElementById('symbol-select').value;
    const timeframe = document.getElementById('timeframe-select').value;
    const btn = document.getElementById('run-analysis-btn');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const progress = document.getElementById('progress-fill');

    btn.disabled = true;
    loading.classList.remove('hidden');
    results.classList.add('hidden');
    progress.style.width = '10%';

    try {
        progress.style.width = '40%';
        const data = await apiPost(`/api/analyze?symbol=${symbol}&timeframe=${timeframe}`);
        progress.style.width = '80%';
        currentResult = data;
        displayResults(data);
        progress.style.width = '100%';
        results.classList.remove('hidden');
        await loadHistory();
    } catch (e) {
        alert('Analysis failed: ' + e.message);
    } finally {
        btn.disabled = false;
        loading.classList.add('hidden');
    }
}

function displayResults(data) {
    // Render charts if candle data available
    if (data._candles && data._candles.length > 0) {
        renderChart(data._candles);
        if (data.layers && data.layers.liquidity) {
            renderChartWithHeatmap(data._candles, data.layers.liquidity);
        }
    }

    // Summary cards
    const dirCard = document.getElementById('direction-card');
    const dirVal = document.getElementById('direction-value');
    dirVal.textContent = data.direction;
    dirCard.className = 'card direction-card ' + data.direction.toLowerCase();

    document.getElementById('confluence-value').textContent = data.confluence_score.toFixed(1) + '/10';
    const confluenceBar = document.getElementById('confluence-bar');
    confluenceBar.style.width = (data.confluence_score / 10 * 100) + '%';
    confluenceBar.style.background = data.confluence_score >= 6 ? 'var(--bullish)' : data.confluence_score >= 4 ? 'var(--neutral)' : 'var(--bearish)';

    document.getElementById('confidence-value').textContent = data.confidence_score.toFixed(0) + '%';
    const gauge = document.getElementById('confidence-gauge');
    gauge.style.width = data.confidence_score + '%';
    gauge.style.background = data.confidence_score >= 75 ? 'var(--bullish)' : data.confidence_score >= 50 ? 'var(--neutral)' : 'var(--bearish)';

    const qualityEl = document.getElementById('quality-value');
    qualityEl.textContent = data.setup_quality;
    qualityEl.className = 'card-value ' + data.setup_quality.toLowerCase() + '-text';

    document.getElementById('price-value').textContent = '$' + data.current_price.toFixed(2);
    const changeEl = document.getElementById('price-change');
    changeEl.textContent = (data.price_change_24h >= 0 ? '+' : '') + data.price_change_24h.toFixed(2) + '% (24h)';
    changeEl.className = 'card-sub ' + (data.price_change_24h >= 0 ? 'bullish-text' : 'bearish-text');

    // AI Summary
    document.getElementById('ai-summary-text').textContent = data.layers.ai_summary.summary;

    // Trade Plan
    displayTradePlan(data.layers.trade_plan);

    // Layers
    displayLayers(data.layers);

    // Heatmap
    displayHeatmap(data.layers);

    // Structure
    displayStructure(data.layers);
}

function displayTradePlan(plan) {
    const section = document.getElementById('trade-plan-section');
    const grid = document.getElementById('trade-plan-grid');
    grid.innerHTML = '';

    if (!plan.valid) {
        section.querySelector('h3').textContent = '📋 Trade Plan — ' + plan.reason;
        const invalid = document.createElement('div');
        invalid.className = 'trade-plan-invalid';
        invalid.textContent = plan.explanation;
        grid.appendChild(invalid);
        return;
    }

    section.querySelector('h3').textContent = '📋 Trade Plan';
    const items = [
        { label: 'Direction', value: plan.direction, cls: plan.direction.toLowerCase() },
        { label: 'Entry', value: '$' + plan.entry.toFixed(4) },
        { label: 'Stop Loss', value: '$' + plan.stop_loss.toFixed(4), cls: 'bearish-text' },
        { label: 'Take Profit 1', value: '$' + plan.take_profit_1.toFixed(4), cls: 'bullish-text' },
        { label: 'Take Profit 2', value: '$' + plan.take_profit_2.toFixed(4), cls: 'bullish-text' },
        { label: 'Invalidation', value: '$' + plan.invalidation.toFixed(4) },
        { label: 'Risk:Reward', value: plan.risk_reward.toFixed(2) + ':1' },
        { label: 'Position Size', value: plan.position_size.toFixed(4) },
        { label: 'Risk %', value: plan.risk_percent + '%' }
    ];

    items.forEach(item => {
        const div = document.createElement('div');
        div.className = 'trade-plan-item';
        div.innerHTML = `<div class="label">${item.label}</div><div class="value ${item.cls || ''}">${item.value}</div>`;
        grid.appendChild(div);
    });
}

function displayLayers(layers) {
    const grid = document.getElementById('layers-grid');
    grid.innerHTML = '';

    const layerNames = {
        trend_range: '1. Trend & Range Engine',
        market_structure: '2. Market Structure',
        liquidity: '3. Liquidity Engine',
        order_blocks: '4. Order Blocks',
        fvg: '5. Fair Value Gap',
        premium_discount: '6. Premium/Discount Zones',
        price_action: '7. Price Action Engine',
        candlestick: '8. Candlestick Scanner',
        chart_pattern: '9. Chart Pattern Scanner',
        volume: '10. Volume Analytics',
        open_interest: '11. Open Interest Analytics',
        funding: '12. Funding Rate Analytics',
        long_short: '13. Long/Short Ratio',
        session: '14. Session Analysis',
        multi_timeframe: '15. Multi-Timeframe Alignment',
        market_regime: '16. Market Regime Engine',
        volatility: '17. Volatility Engine',
        rmt: '18. RMT Analytics Engine',
        confluence: '19. Confluence Score',
        setup_quality: '20. Setup Quality Engine',
        confidence: '21. Confidence Meter',
        trade_plan: '23. Trade Plan Generator',
        ai_summary: '24. AI Summary'
    };

    for (const [key, name] of Object.entries(layerNames)) {
        const layer = layers[key];
        if (!layer) continue;

        const card = document.createElement('div');
        card.className = 'layer-card';
        const score = layer.score !== undefined ? layer.score : (layer.confluence_score !== undefined ? layer.confluence_score / 10 : 0.5);
        const scoreClass = score >= 0.7 ? 'high' : score >= 0.4 ? 'medium' : 'low';
        const scorePct = Math.round(score * 100);

        let details = '';
        if (layer.patterns && layer.patterns.length > 0) {
            details = layer.patterns.slice(0, 5).map(p => p.pattern || JSON.stringify(p)).join(', ');
        } else if (layer.blocks && layer.blocks.length > 0) {
            details = layer.blocks.length + ' blocks';
        } else if (layer.fvgs && layer.fvgs.length > 0) {
            details = layer.fvgs.length + ' FVGs';
        } else if (layer.pools && layer.pools.length > 0) {
            details = layer.pools.length + ' pools';
        } else if (layer.sr_levels && layer.sr_levels.length > 0) {
            details = layer.sr_levels.length + ' S/R levels';
        } else if (layer.trend) {
            details = 'Trend: ' + layer.trend;
        } else if (layer.structure) {
            details = 'Structure: ' + layer.structure;
        } else if (layer.regime) {
            details = 'Regime: ' + layer.regime;
        } else if (layer.sentiment) {
            details = 'Sentiment: ' + layer.sentiment;
        } else if (layer.ratio) {
            details = 'L/S Ratio: ' + layer.ratio;
        } else if (layer.alignment) {
            details = 'Alignment: ' + layer.alignment;
        } else if (layer.eigenvalues) {
            details = 'Signals: ' + layer.signals + ', Noise: ' + layer.noise_components;
        }

        card.innerHTML = `
            <div class="layer-header">
                <span class="layer-name">${name}</span>
                <span class="layer-score ${scoreClass}">${scorePct}%</span>
            </div>
            <div class="layer-explanation">${layer.explanation || ''}</div>
            ${details ? `<div class="layer-details">${details}</div>` : ''}
        `;
        grid.appendChild(card);
    }
}

function displayHeatmap(layers) {
    const container = document.getElementById('heatmap-container');
    container.innerHTML = '';

    const liquidity = layers.liquidity || {};
    const pools = liquidity.pools || [];
    const sweeps = liquidity.sweeps || [];

    if (pools.length === 0) {
        container.innerHTML = '<p style="color:var(--text-secondary)">No significant liquidity pools detected.</p>';
        return;
    }

    // Group by level and show heat intensity
    const levels = {};
    pools.forEach(p => {
        const key = p.level.toFixed(2);
        if (!levels[key]) levels[key] = { level: p.level, strength: 0, type: p.type };
        levels[key].strength += p.strength || 1;
    });

    const maxStrength = Math.max(...Object.values(levels).map(l => l.strength), 1);

    Object.values(levels).sort((a, b) => b.strength - a.strength).slice(0, 15).forEach(l => {
        const bar = document.createElement('div');
        bar.className = 'heatmap-bar';
        const pct = (l.strength / maxStrength * 100);
        const isHot = l.type === 'EQUAL_HIGH';
        bar.innerHTML = `
            <div class="heatmap-label">${l.type === 'EQUAL_HIGH' ? 'Resistance' : 'Support'} $${l.level.toFixed(2)}</div>
            <div class="heatmap-track">
                <div class="heatmap-fill ${isHot ? 'hot' : 'cold'}" style="width:${pct}%"></div>
            </div>
            <div class="heatmap-value">${l.strength}x</div>
        `;
        container.appendChild(bar);
    });

    if (sweeps.length > 0) {
        const sweepNote = document.createElement('div');
        sweepNote.style.cssText = 'margin-top:12px;font-size:0.8rem;color:var(--neutral);';
        sweepNote.textContent = `⚡ ${sweeps.length} recent liquidity sweep(s) detected`;
        container.appendChild(sweepNote);
    }
}

function displayStructure(layers) {
    const visual = document.getElementById('structure-visual');
    const ms = layers.market_structure || {};
    const hhhl = ms.hh_hl_lh_ll || [];
    const bos = ms.bos || [];
    const choch = ms.choch || [];

    let text = 'MARKET STRUCTURE ANALYSIS\n';
    text += '═'.repeat(50) + '\n\n';
    text += `Current Structure: ${ms.structure || 'UNDEFINED'}\n`;
    text += `Structure Shifts: ${ms.shifts || 0}\n\n`;

    if (hhhl.length > 0) {
        text += 'Recent Swing Points: ' + hhhl.slice(-10).join(' → ') + '\n\n';
    }

    if (bos.length > 0) {
        text += 'BREAK OF STRUCTURE (BOS):\n';
        bos.forEach(b => {
            text += `  [${b.type}] Level: $${b.level.toFixed(2)}\n`;
        });
        text += '\n';
    }

    if (choch.length > 0) {
        text += 'CHANGE OF CHARACTER (CHoCH):\n';
        choch.forEach(c => {
            text += `  [${c.type}] Level: $${c.level.toFixed(2)}\n`;
        });
        text += '\n';
    }

    text += '═'.repeat(50) + '\n';
    text += 'BOS = Break of Structure (trend continuation)\n';
    text += 'CHoCH = Change of Character (potential reversal)\n';
    text += 'HH = Higher High | HL = Higher Low | LH = Lower High | LL = Lower Low';

    visual.textContent = text;
}


// ===== CHART RENDERING =====
let chartInstance = null;
let chartHeatmapInstance = null;

function renderChart(candles) {
    const container = document.getElementById('chart-container');
    if (!container || !candles || candles.length === 0) return;

    container.innerHTML = '';

    const chart = LightweightCharts.createChart(container, {
        layout: {
            background: { color: '#1a1f2e' },
            textColor: '#94a3b8',
        },
        grid: {
            vertLines: { color: '#2d3748' },
            horzLines: { color: '#2d3748' },
        },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        rightPriceScale: { borderColor: '#2d3748' },
        timeScale: { borderColor: '#2d3748' },
        width: container.clientWidth,
        height: 400,
    });

    const candleSeries = chart.addCandlestickSeries({
        upColor: '#10b981',
        downColor: '#ef4444',
        borderUpColor: '#10b981',
        borderDownColor: '#ef4444',
        wickUpColor: '#10b981',
        wickDownColor: '#ef4444',
    });

    const chartData = candles.map(c => ({
        time: c.time / 1000,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
    }));

    candleSeries.setData(chartData);
    chart.timeScale().fitContent();

    chartInstance = chart;
}

function renderChartWithHeatmap(candles, liquidityData) {
    const container = document.getElementById('chart-heatmap-container');
    const overlay = document.getElementById('heatmap-overlay');
    if (!container || !candles || candles.length === 0) return;

    container.innerHTML = '';
    overlay.innerHTML = '';

    const chart = LightweightCharts.createChart(container, {
        layout: {
            background: { color: '#1a1f2e' },
            textColor: '#94a3b8',
        },
        grid: {
            vertLines: { color: '#2d3748' },
            horzLines: { color: '#2d3748' },
        },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        rightPriceScale: { borderColor: '#2d3748' },
        timeScale: { borderColor: '#2d3748' },
        width: container.clientWidth,
        height: 400,
    });

    const candleSeries = chart.addCandlestickSeries({
        upColor: '#10b981',
        downColor: '#ef4444',
        borderUpColor: '#10b981',
        borderDownColor: '#ef4444',
        wickUpColor: '#10b981',
        wickDownColor: '#ef4444',
    });

    const chartData = candles.map(c => ({
        time: c.time / 1000,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
    }));

    candleSeries.setData(chartData);

    // Add liquidity zones as horizontal lines with heat intensity
    if (liquidityData && liquidityData.pools) {
        const pools = liquidityData.pools.slice(0, 10);
        const maxStrength = Math.max(...pools.map(p => p.strength || 1), 1);

        pools.forEach(pool => {
            const intensity = (pool.strength || 1) / maxStrength;
            const color = pool.type === 'EQUAL_HIGH' 
                ? `rgba(239, 68, 68, ${0.3 + intensity * 0.5})`  // Red for resistance
                : `rgba(16, 185, 129, ${0.3 + intensity * 0.5})`; // Green for support

            const line = chart.addLineSeries({
                color: color,
                lineWidth: 1 + intensity * 3,
                lastValueVisible: false,
                title: `${pool.type} $${pool.level.toFixed(2)}`,
            });

            line.setData(chartData.map(d => ({
                time: d.time,
                value: pool.level,
            })));
        });

        // Add sweep markers
        if (liquidityData.sweeps) {
            liquidityData.sweeps.forEach(sweep => {
                const markerTime = candles[sweep.index]?.time / 1000;
                if (markerTime) {
                    candleSeries.setMarkers([{
                        time: markerTime,
                        position: sweep.type === 'HIGH_SWEEP' ? 'aboveBar' : 'belowBar',
                        color: sweep.type === 'HIGH_SWEEP' ? '#ef4444' : '#10b981',
                        shape: sweep.type === 'HIGH_SWEEP' ? 'arrowDown' : 'arrowUp',
                        text: 'SWEEP',
                    }]);
                }
            });
        }
    }

    // Add order blocks as colored zones
    if (liquidityData && liquidityData.blocks) {
        const freshBlocks = (liquidityData.blocks || []).filter(b => !b.mitigated).slice(0, 5);
        freshBlocks.forEach(block => {
            const color = block.type === 'BULLISH' 
                ? 'rgba(16, 185, 129, 0.15)' 
                : 'rgba(239, 68, 68, 0.15)';

            // Add as a very thin line series to simulate zone
            const zoneLine = chart.addLineSeries({
                color: color,
                lineWidth: 8,
                lastValueVisible: false,
            });

            zoneLine.setData(chartData.map(d => ({
                time: d.time,
                value: (block.top + block.bottom) / 2,
            })));
        });
    }

    chart.timeScale().fitContent();
    chartHeatmapInstance = chart;
}
// ===== BACKTEST =====
async function runBacktest() {
    const symbol = document.getElementById('symbol-select').value;
    const timeframe = document.getElementById('timeframe-select').value;
    const btn = document.getElementById('backtest-btn');
    const section = document.getElementById('backtest-section');

    btn.disabled = true;
    btn.textContent = '⏳ Backtesting...';
    section.classList.remove('hidden');

    try {
        const data = await apiPost(`/api/backtest?symbol=${symbol}&timeframe=${timeframe}&days=90`);
        displayBacktest(data);
    } catch (e) {
        alert('Backtest failed: ' + e.message);
        section.classList.add('hidden');
    } finally {
        btn.disabled = false;
        btn.textContent = '📊 Backtest';
    }
}

function displayBacktest(data) {
    const grid = document.getElementById('backtest-grid');
    grid.innerHTML = '';

    const items = [
        { label: 'Total Trades', value: data.total_trades },
        { label: 'Win Rate', value: data.win_rate.toFixed(1) + '%', cls: data.win_rate >= 50 ? 'bullish-text' : 'bearish-text' },
        { label: 'Profit Factor', value: data.profit_factor.toFixed(2), cls: data.profit_factor >= 1.5 ? 'bullish-text' : 'bearish-text' },
        { label: 'Avg R:R', value: data.avg_rr.toFixed(2) + ':1' },
        { label: 'Period', value: data.start_date.split('T')[0] + ' to ' + data.end_date.split('T')[0] },
        { label: 'Strategy', value: data.strategy }
    ];

    items.forEach(item => {
        const div = document.createElement('div');
        div.className = 'trade-plan-item';
        div.innerHTML = `<div class="label">${item.label}</div><div class="value ${item.cls || ''}">${item.value}</div>`;
        grid.appendChild(div);
    });

    // Trades table
    const tableDiv = document.getElementById('trades-table');
    if (data.trades && data.trades.length > 0) {
        let html = '<table><thead><tr><th>#</th><th>Type</th><th>Entry</th><th>Exit</th><th>P&L</th><th>Result</th></tr></thead><tbody>';
        data.trades.forEach((t, i) => {
            html += `<tr>
                <td>${i+1}</td>
                <td>${t.type}</td>
                <td>$${t.entry.toFixed(4)}</td>
                <td>$${t.exit.toFixed(4)}</td>
                <td class="${t.result === 'WIN' ? 'win' : 'loss'}">${(t.pnl * 100).toFixed(2)}%</td>
                <td class="${t.result === 'WIN' ? 'win' : 'loss'}">${t.result}</td>
            </tr>`;
        });
        html += '</tbody></table>';
        tableDiv.innerHTML = html;
    } else {
        tableDiv.innerHTML = '<p style="color:var(--text-secondary)">No trades in backtest period.</p>';
    }
}

// ===== SCANNER =====
async function runScanner() {
    const btn = document.getElementById('scanner-btn');
    const section = document.getElementById('scanner-section');
    const table = document.getElementById('scanner-table');

    btn.disabled = true;
    btn.textContent = '⏳ Scanning...';
    section.classList.remove('hidden');
    table.innerHTML = '<p style="color:var(--text-secondary)">Scanning watchlist across timeframes...</p>';

    const results = [];
    const timeframes = ['1h', '4h', '1d'];

    for (const sym of watchlist.slice(0, 5)) {  // Limit to 5 for speed
        for (const tf of ['1h']) {  // Primary TF only for speed
            try {
                const data = await apiPost(`/api/analyze?symbol=${sym}&timeframe=${tf}`);
                results.push({
                    symbol: sym,
                    timeframe: tf,
                    direction: data.direction,
                    confluence: data.confluence_score,
                    confidence: data.confidence_score,
                    quality: data.setup_quality,
                    price: data.current_price,
                    change: data.price_change_24h
                });
            } catch (e) {
                results.push({ symbol: sym, timeframe: tf, error: true });
            }
        }
    }

    displayScanner(results);
    btn.disabled = false;
    btn.textContent = '🔍 Scan Watchlist';
}

function displayScanner(results) {
    const table = document.getElementById('scanner-table');
    if (results.length === 0) {
        table.innerHTML = '<p style="color:var(--text-secondary)">No results.</p>';
        return;
    }

    let html = '<table><thead><tr><th>Symbol</th><th>TF</th><th>Direction</th><th>Confluence</th><th>Confidence</th><th>Quality</th><th>Price</th><th>24h %</th></tr></thead><tbody>';
    results.forEach(r => {
        if (r.error) {
            html += `<tr><td>${r.symbol}</td><td colspan="7" style="color:var(--bearish)">Error fetching data</td></tr>`;
            return;
        }
        const dirClass = r.direction === 'BULLISH' ? 'bullish-text' : r.direction === 'BEARISH' ? 'bearish-text' : 'neutral-text';
        const qualClass = r.quality === 'ELITE' ? 'elite-text' : r.quality === 'STRONG' ? 'strong-text' : r.quality === 'GOOD' ? 'good-text' : r.quality === 'MODERATE' ? 'moderate-text' : 'weak-text';
        html += `<tr>
            <td><strong>${r.symbol}</strong></td>
            <td>${r.timeframe}</td>
            <td class="${dirClass}">${r.direction}</td>
            <td>${r.confluence.toFixed(1)}/10</td>
            <td>${r.confidence.toFixed(0)}%</td>
            <td class="${qualClass}">${r.quality}</td>
            <td>$${r.price.toFixed(2)}</td>
            <td class="${r.change >= 0 ? 'bullish-text' : 'bearish-text'}">${r.change >= 0 ? '+' : ''}${r.change.toFixed(2)}%</td>
        </tr>`;
    });
    html += '</tbody></table>';
    table.innerHTML = html;
}

// ===== HISTORY =====
async function loadHistory() {
    try {
        const data = await apiGet('/api/history?limit=10');
        displayHistory(data.history);
    } catch (e) {
        console.error('History load error:', e);
    }
}

function displayHistory(history) {
    const table = document.getElementById('history-table');
    if (!history || history.length === 0) {
        table.innerHTML = '<p style="color:var(--text-secondary)">No analysis history yet.</p>';
        return;
    }

    let html = '<table><thead><tr><th>Time</th><th>Symbol</th><th>TF</th><th>Direction</th><th>Confluence</th><th>Confidence</th><th>Quality</th></tr></thead><tbody>';
    history.forEach(h => {
        const r = h.result || h;
        const dirClass = r.direction === 'BULLISH' ? 'bullish-text' : r.direction === 'BEARISH' ? 'bearish-text' : 'neutral-text';
        const qual = r.setup_quality || r.quality || 'N/A';
        const qualClass = qual === 'ELITE' ? 'elite-text' : qual === 'STRONG' ? 'strong-text' : qual === 'GOOD' ? 'good-text' : qual === 'MODERATE' ? 'moderate-text' : 'weak-text';
        const time = h.timestamp ? h.timestamp.split('T')[0] + ' ' + h.timestamp.split('T')[1].split('.')[0].substring(0,5) : 'N/A';
        html += `<tr>
            <td>${time}</td>
            <td><strong>${h.symbol || r.symbol}</strong></td>
            <td>${h.timeframe || r.timeframe}</td>
            <td class="${dirClass}">${r.direction}</td>
            <td>${(r.confluence_score || 0).toFixed(1)}/10</td>
            <td>${(r.confidence_score || 0).toFixed(0)}%</td>
            <td class="${qualClass}">${qual}</td>
        </tr>`;
    });
    html += '</tbody></table>';
    table.innerHTML = html;
}

// ===== KEYBOARD SHORTCUTS =====
document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && document.activeElement.id === 'new-symbol') {
        addToWatchlist();
    }
    if (e.key === 'F5' || (e.ctrlKey && e.key === 'r')) {
        e.preventDefault();
        runAnalysis();
    }
});
