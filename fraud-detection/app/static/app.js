/**
 * FraudShield — Dashboard Interactivity
 * Real-time transaction simulation and model comparison
 */

// State
let totalProcessed = 0;
let fraudDetected = 0;
let autoSimulating = false;
let autoInterval = null;

// Elements
const totalEl = document.getElementById('totalProcessed');
const fraudEl = document.getElementById('fraudDetected');
const speedSlider = document.getElementById('simSpeed');
const speedLabel = document.getElementById('speedLabel');
const feedContainer = document.getElementById('transactionFeed');
const decisionPanel = document.getElementById('decisionPanel');
const btnAuto = document.getElementById('btnAutoSimulate');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadStats();
    speedSlider.addEventListener('input', () => {
        const val = speedSlider.value / 1000;
        speedLabel.textContent = val.toFixed(1) + 's';
        if (autoSimulating) {
            clearInterval(autoInterval);
            autoInterval = setInterval(simulateTransaction, parseInt(speedSlider.value));
        }
    });
});

/**
 * Load model evaluation and benchmark stats
 */
async function loadStats() {
    try {
        const res = await fetch('/api/stats');
        const data = await res.json();

        if (data.evaluation) {
            updateModelCard('xgb', data.evaluation.XGBoost);
            updateModelCard('lgb', data.evaluation.LightGBM);
            updateModelCard('lstm', data.evaluation.LSTM);
        }

        if (data.benchmarks) {
            data.benchmarks.forEach(b => {
                const prefix = b.model === 'XGBoost' ? 'xgb' :
                               b.model === 'LightGBM' ? 'lgb' : 'lstm';
                updateLatency(prefix, b.p95_ms);
            });
        }
    } catch (e) {
        console.log('Stats not available yet:', e.message);
    }
}

/**
 * Update a model card with evaluation metrics
 */
function updateModelCard(prefix, metrics) {
    if (!metrics) return;

    animateValue(`${prefix}-auc`, parseFloat(metrics.auc_roc).toFixed(4));
    animateValue(`${prefix}-precision`, parseFloat(metrics.precision).toFixed(4));
    animateValue(`${prefix}-recall`, parseFloat(metrics.recall).toFixed(4));
    animateValue(`${prefix}-f1`, parseFloat(metrics.f1).toFixed(4));
}

/**
 * Update latency display for a model
 */
function updateLatency(prefix, p95Ms) {
    const bar = document.getElementById(`${prefix}-latency-bar`);
    const val = document.getElementById(`${prefix}-latency`);

    if (!bar || !val) return;

    const pct = Math.min((p95Ms / 5.0) * 100, 100); // 5ms = 100%
    bar.style.width = pct + '%';

    if (p95Ms > 3.0) {
        bar.className = 'latency-fill danger';
        val.style.color = 'var(--accent-red)';
    } else if (p95Ms > 2.0) {
        bar.className = 'latency-fill warning';
        val.style.color = 'var(--accent-orange)';
    } else {
        bar.className = 'latency-fill';
        val.style.color = 'var(--accent-green)';
    }

    val.textContent = p95Ms.toFixed(3) + 'ms';
}

/**
 * Animate text value change
 */
function animateValue(id, newValue) {
    const el = document.getElementById(id);
    if (!el) return;
    el.style.transition = 'opacity 0.2s';
    el.style.opacity = '0';
    setTimeout(() => {
        el.textContent = newValue;
        el.style.opacity = '1';
    }, 200);
}

/**
 * Simulate a single transaction
 */
async function simulateTransaction() {
    const btn = document.getElementById('btnSimulate');
    btn.disabled = true;

    try {
        const res = await fetch('/api/simulate');
        const data = await res.json();

        totalProcessed++;
        if (data.consensus.is_fraud) fraudDetected++;

        totalEl.textContent = totalProcessed.toLocaleString();
        fraudEl.textContent = fraudDetected.toLocaleString();

        updateDecisionPanel(data);
        addToFeed(data);
    } catch (e) {
        console.error('Simulation error:', e);
        showError();
    } finally {
        btn.disabled = false;
    }
}

/**
 * Toggle auto-simulate mode
 */
function toggleAutoSimulate() {
    autoSimulating = !autoSimulating;

    if (autoSimulating) {
        btnAuto.classList.add('active');
        btnAuto.querySelector('.btn-icon').textContent = '⏸';
        autoInterval = setInterval(simulateTransaction, parseInt(speedSlider.value));
    } else {
        btnAuto.classList.remove('active');
        btnAuto.querySelector('.btn-icon').textContent = '⟳';
        clearInterval(autoInterval);
    }
}

/**
 * Update the decision panel with prediction results
 */
function updateDecisionPanel(data) {
    const txn = data.transaction;
    const consensus = data.consensus;
    const models = data.models;

    const isFraud = consensus.is_fraud;

    decisionPanel.innerHTML = `
        <div class="decision-result">
            <div class="decision-verdict">
                <span class="verdict-icon">${isFraud ? '🚨' : '✅'}</span>
                <span class="verdict-text ${isFraud ? 'block' : 'approve'}">
                    ${consensus.decision}
                </span>
            </div>
            <div class="decision-details">
                <div class="detail-item">
                    <div class="detail-label">Amount</div>
                    <div class="detail-value">$${txn.amount.toLocaleString()}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Card</div>
                    <div class="detail-value">****${txn.card_last4}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Category</div>
                    <div class="detail-value">${formatCategory(txn.category)}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Risk Score</div>
                    <div class="detail-value" style="color: ${isFraud ? 'var(--accent-red)' : 'var(--accent-green)'}">
                        ${(consensus.probability * 100).toFixed(1)}%
                    </div>
                </div>
            </div>
            <div class="decision-models">
                ${renderModelResult('XGBoost', models.xgboost)}
                ${renderModelResult('LightGBM', models.lightgbm)}
                ${renderModelResult('LSTM', models.lstm)}
            </div>
        </div>
    `;
}

/**
 * Render a single model result bar
 */
function renderModelResult(name, result) {
    const prob = result.probability;
    const pct = prob * 100;
    const color = prob > 0.5 ? 'var(--accent-red)' : prob > 0.3 ? 'var(--accent-orange)' : 'var(--accent-green)';

    return `
        <div class="model-result">
            <span class="model-result-name">${name}</span>
            <div class="model-result-bar">
                <div class="model-result-bar-fill" style="width: ${pct}%; background: ${color};"></div>
            </div>
            <span class="model-result-value" style="color: ${color}">${(pct).toFixed(1)}%</span>
            <span class="model-result-latency">${result.latency_ms.toFixed(2)}ms</span>
        </div>
    `;
}

/**
 * Add a transaction to the live feed
 */
function addToFeed(data) {
    const txn = data.transaction;
    const consensus = data.consensus;
    const isFraud = consensus.is_fraud;
    const avgLatency = (
        data.models.xgboost.latency_ms +
        data.models.lightgbm.latency_ms +
        data.models.lstm.latency_ms
    ) / 3;

    // Remove empty state
    const empty = feedContainer.querySelector('.feed-empty');
    if (empty) empty.remove();

    const item = document.createElement('div');
    item.className = `feed-item ${isFraud ? 'fraud-alert' : ''}`;
    item.innerHTML = `
        <div class="feed-indicator ${isFraud ? 'fraud' : 'safe'}"></div>
        <div class="feed-info">
            <div class="feed-merchant">${txn.merchant}</div>
            <div class="feed-meta">${formatCategory(txn.category)} · ${formatHour(txn.hour)} · ****${txn.card_last4}</div>
        </div>
        <div class="feed-amount" style="color: ${isFraud ? 'var(--accent-red)' : 'var(--text-primary)'}">
            $${txn.amount.toLocaleString()}
        </div>
        <div class="feed-probability" style="color: ${isFraud ? 'var(--accent-red)' : 'var(--accent-green)'}">
            ${(consensus.probability * 100).toFixed(1)}%
        </div>
        <div class="feed-decision ${isFraud ? 'block' : 'approve'}">
            ${consensus.decision}
        </div>
        <div class="feed-latency">${avgLatency.toFixed(2)}ms</div>
    `;

    feedContainer.insertBefore(item, feedContainer.firstChild);

    // Keep max 50 items
    while (feedContainer.children.length > 50) {
        feedContainer.removeChild(feedContainer.lastChild);
    }
}

/**
 * Format merchant category for display
 */
function formatCategory(cat) {
    const map = {
        'grocery': '🛒 Grocery',
        'gas_station': '⛽ Gas Station',
        'restaurant': '🍽️ Restaurant',
        'online': '🌐 Online',
        'retail': '🏬 Retail',
        'travel': '✈️ Travel',
        'entertainment': '🎬 Entertainment',
    };
    return map[cat] || cat;
}

/**
 * Format hour for display
 */
function formatHour(hour) {
    const period = hour >= 12 ? 'PM' : 'AM';
    const h = hour % 12 || 12;
    return `${h}:00 ${period}`;
}

/**
 * Show error state in decision panel
 */
function showError() {
    decisionPanel.innerHTML = `
        <div class="decision-waiting" style="color: var(--accent-red);">
            <span>⚠️ Connection error. Make sure the server is running.</span>
        </div>
    `;
}
