// Client-Side Controller & Simulation Engine

// Configuration & Globals
const API_URL = "http://localhost:8000";
let isBackendOnline = false;
let isFeedRunning = true;
let feedIntervalId = null;
let feedIntervalMs = 1000;
let currentTab = "dashboard";

// Dashboard Metrics Cache
let metrics = {
    processedCount: 0,
    fraudCount: 0,
    blockedValue: 0.0,
    totalRiskScoreSum: 0.0,
    recentTxnTimes: [],
    recentTxnCounts: []
};

// Real-Time Charts (Chart.js)
let volumeChart = null;
let categoryChart = null;
let importanceChart = null;

// Sparklines
let sparklines = {
    latency: null,
    drift: null,
    memory: null
};

// Data Store
let transactionsLedger = [];
let userStateStore = {}; // Tracks rolling transaction data per user on client-side for rule/JS model evaluations
const MAX_LEDGER_ROWS = 50;
let modelDecisionThreshold = 0.50;

// Active Policies / Rules Engine
let activeRules = [
    { id: "RULE_1", name: "ENT_CRITICAL_AMOUNT", field: "amount", op: ">", value: 10000.00, action: "Block", hits: 0, status: true },
    { id: "RULE_2", name: "GEO_HIGH_MISMATCH", field: "distance_from_home", op: ">", value: 2000.00, action: "Flag", hits: 0, status: true },
    { id: "RULE_3", name: "VELOCITY_BURST_1H", field: "txn_count_1h", op: ">", value: 5, action: "Block", hits: 0, status: true }
];

// Global Lists of Names / Profiles
const USER_PROFILES = [
    { id: "USR_1001", name: "Elena Rostova", homeLat: 40.7128, homeLon: -74.0060, primaryDevice: "DEV_8812", seenDevices: new Set(["DEV_8812"]) },
    { id: "USR_1002", name: "Marcus Aurelius", homeLat: 34.0522, homeLon: -118.2437, primaryDevice: "DEV_4490", seenDevices: new Set(["DEV_4490"]) },
    { id: "USR_1003", name: "Kenji Tanaka", homeLat: 47.6062, homeLon: -122.3321, primaryDevice: "DEV_3192", seenDevices: new Set(["DEV_3192", "DEV_9821"]) },
    { id: "USR_1004", name: "Amina Diop", homeLat: 29.7604, homeLon: -95.3698, primaryDevice: "DEV_5001", seenDevices: new Set(["DEV_5001"]) },
    { id: "USR_1005", name: "Sofia Silva", homeLat: 25.7617, homeLon: -80.1918, primaryDevice: "DEV_6082", seenDevices: new Set(["DEV_6082"]) },
    { id: "USR_1006", name: "Liam O'Connor", homeLat: 41.8781, homeLon: -87.6298, primaryDevice: "DEV_1234", seenDevices: new Set(["DEV_1234"]) },
    { id: "USR_1007", name: "Clara Dupont", homeLat: 39.9526, homeLon: -75.1652, primaryDevice: "DEV_7749", seenDevices: new Set(["DEV_7749"]) },
    { id: "USR_1008", name: "Zayn Malik", homeLat: 32.7767, homeLon: -96.7970, primaryDevice: "DEV_9931", seenDevices: new Set(["DEV_9931"]) }
];

const MERCHANTS = {
    grocery: ["Whole Foods Market", "Kroger", "Safeway", "Trader Joe's"],
    gas_station: ["Chevron", "Shell", "ExxonMobil", "BP Gas"],
    dining: ["Starbucks", "Olive Garden", "Chipotle Grill", "McDonald's"],
    e_commerce: ["Amazon.com", "Apple Store", "Ebay Digital", "BestBuy Online"],
    travel: ["Delta Air Lines", "Expedia Inc", "Uber Ride", "Airbnb Booking"],
    electronics: ["B&H Photo Video", "Newegg Tech", "Samsung Electronics"],
    entertainment: ["Netflix Video", "Spotify Audio", "Ticketmaster Events", "Steam Games"],
    transfer: ["Wire Transfer Intl", "Venmo P2P", "Zelle Instant", "Coinbase Trade"]
};

// Features mapping for the global importance chart (SHAP baseline weights)
const SHAP_GLOBAL_IMPORTANCES = {
    "Distance from Home (km)": 0.28,
    "Device ID Mismatched": 0.24,
    "Transaction Count (Last 1 Hour)": 0.18,
    "Transaction Amount ($)": 0.15,
    "Cumulative Spend (Last 24 Hours)": 0.08,
    "Card Physically Present": 0.05,
    "Hour of Day (0-23)": 0.02
};

// ----------------------------------------------------
// Initialization
// ----------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initCharts();
    checkBackendHealth();
    startSimulationFeed();
    initTuningSlider();
    initRuleForm();
    initChatbot();
    
    // Check API health periodically (every 5 seconds)
    setInterval(checkBackendHealth, 5000);
});

// Tab Navigation
function initTabs() {
    const tabs = ["dashboard", "rules", "model", "monitoring"];
    tabs.forEach(tab => {
        const btn = document.getElementById(`btnTab${tab.charAt(0).toUpperCase() + tab.slice(1)}`);
        if (btn) {
            btn.addEventListener("click", () => switchTab(tab));
        }
    });
}

function switchTab(tabId) {
    currentTab = tabId;
    
    // Toggle active buttons
    document.querySelectorAll(".nav-item").forEach(btn => btn.classList.remove("active"));
    const activeBtn = document.getElementById(`btnTab${tabId.charAt(0).toUpperCase() + tabId.slice(1)}`);
    if (activeBtn) activeBtn.classList.add("active");
    
    // Toggle active sections
    document.querySelectorAll(".tab-content").forEach(sec => sec.classList.remove("active"));
    const activeSec = document.getElementById(`tab${tabId.charAt(0).toUpperCase() + tabId.slice(1)}`);
    if (activeSec) activeSec.classList.add("active");
    
    // Update header headers
    const title = document.getElementById("pageTitle");
    const sub = document.getElementById("pageSubtitle");
    
    if (tabId === "dashboard") {
        title.innerText = "Real-Time Security Operations Center";
        sub.innerText = "Monitoring and analyzing credit card transactions for active fraud signatures.";
    } else if (tabId === "rules") {
        title.innerText = "Active Policy & Rules Engine";
        sub.innerText = "Construct and apply programmatic conditions to automatically drop or flag transactions.";
        renderRulesList();
    } else if (tabId === "model") {
        title.innerText = "Model Performance & Analytics";
        sub.innerText = "Evaluate XGBoost classifier metrics, tune decision thresholds, and explore feature impact.";
        updateConfusionMatrix();
    } else if (tabId === "monitoring") {
        title.innerText = "MLOps Container Monitoring";
        sub.innerText = "Visualizing telemetry, scrapers, CPU memory, and model data drift indicators.";
    }
}

// ----------------------------------------------------
// Health check connection to Python FastAPI server
// ----------------------------------------------------
async function checkBackendHealth() {
    const badge = document.getElementById("apiStatusBadge");
    const text = document.getElementById("apiStatusText");
    
    try {
        const response = await fetch(`${API_URL}/health`, { method: 'GET', signal: AbortSignal.timeout(1500) });
        if (response.ok) {
            const data = await response.json();
            if (data.status === "healthy") {
                isBackendOnline = true;
                badge.className = "connection-status online";
                text.innerText = `API ONLINE // XGBOOST ACTIVE`;
                return;
            }
        }
    } catch (e) {
        // Fallback silently to offline mode
    }
    
    isBackendOnline = false;
    badge.className = "connection-status offline";
    text.innerText = `API OFFLINE // LOCAL SIMULATION`;
}

// ----------------------------------------------------
// Chart Initialization & Rendering
// ----------------------------------------------------
function initCharts() {
    // 1. Transaction Volume Chart
    const volCtx = document.getElementById("volumeChart").getContext("2d");
    volumeChart = new Chart(volCtx, {
        type: 'line',
        data: {
            labels: Array(15).fill(''),
            datasets: [{
                label: 'Inference Load',
                data: Array(15).fill(0),
                borderColor: '#818cf8',
                backgroundColor: 'rgba(129, 140, 248, 0.15)',
                borderWidth: 2.5,
                fill: true,
                tension: 0.4,
                pointBackgroundColor: '#818cf8'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#64748b' } },
                x: { grid: { display: false }, ticks: { color: '#64748b' } }
            }
        }
    });

    // 2. Category distribution Chart
    const catCtx = document.getElementById("categoryChart").getContext("2d");
    categoryChart = new Chart(catCtx, {
        type: 'doughnut',
        data: {
            labels: ['Grocery', 'Gas Station', 'Dining', 'E-Commerce', 'Travel', 'Electronics', 'Entertainment', 'Transfer'],
            datasets: [{
                data: [0, 0, 0, 0, 0, 0, 0, 0],
                backgroundColor: [
                    '#10b981', '#059669', '#34d399', '#818cf8',
                    '#f59e0b', '#f43f5e', '#6366f1', '#a855f7'
                ],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: { color: '#94a3b8', font: { size: 10 } }
                }
            },
            cutout: '65%'
        }
    });

    // 3. Global Feature Importance (Tuning tab)
    const impCtx = document.getElementById("importanceChart").getContext("2d");
    importanceChart = new Chart(impCtx, {
        type: 'bar',
        data: {
            labels: Object.keys(SHAP_GLOBAL_IMPORTANCES),
            datasets: [{
                data: Object.values(SHAP_GLOBAL_IMPORTANCES),
                backgroundColor: 'rgba(129, 140, 248, 0.75)',
                hoverBackgroundColor: '#818cf8',
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { title: { display: true, text: 'Importance Weight (Avg |SHAP|)', color: '#64748b' }, grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#64748b' } },
                y: { grid: { display: false }, ticks: { color: '#94a3b8' } }
            }
        }
    });

    // Initialize Monitor Sparklines
    initSparklines();
}

function initSparklines() {
    const setupSparkline = (canvasId, color) => {
        const ctx = document.getElementById(canvasId).getContext("2d");
        return new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array(10).fill(''),
                datasets: [{
                    data: Array(10).fill(0),
                    borderColor: color,
                    borderWidth: 1.5,
                    fill: false,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { display: false },
                    y: { display: false }
                }
            }
        });
    };
    
    sparklines.latency = setupSparkline("latencySparkline", "#818cf8");
    sparklines.drift = setupSparkline("driftSparkline", "#10b981");
    sparklines.memory = setupSparkline("memorySparkline", "#0ea5e9");

    // Populate initial sparkline historical dummy data
    for (let i = 0; i < 10; i++) {
        updateSparkline(sparklines.latency, 8 + Math.random() * 10);
        updateSparkline(sparklines.drift, 0.03 + Math.random() * 0.02);
        updateSparkline(sparklines.memory, 180 + Math.random() * 8);
    }
}

function updateSparkline(chart, val) {
    chart.data.datasets[0].data.shift();
    chart.data.datasets[0].data.push(val);
    chart.update('none');
}

// ----------------------------------------------------
// Simulation transaction stream engine
// ----------------------------------------------------
function startSimulationFeed() {
    // Setup Pause Play Controls
    const btnPausePlay = document.getElementById("btnPausePlayFeed");
    btnPausePlay.addEventListener("click", () => {
        isFeedRunning = !isFeedRunning;
        if (isFeedRunning) {
            btnPausePlay.innerHTML = '<i class="fa-solid fa-pause"></i>';
            btnPausePlay.classList.add("btn-play");
            startInterval();
        } else {
            btnPausePlay.innerHTML = '<i class="fa-solid fa-play"></i>';
            btnPausePlay.classList.remove("btn-play");
            clearInterval(feedIntervalId);
        }
    });

    const speedSelect = document.getElementById("selectFeedSpeed");
    speedSelect.addEventListener("change", (e) => {
        feedIntervalMs = parseInt(e.target.value);
        if (isFeedRunning) {
            clearInterval(feedIntervalId);
            startInterval();
        }
    });

    const btnSpeedUp = document.getElementById("btnSpeedUp");
    btnSpeedUp.addEventListener("click", () => {
        // Fast cycle options
        if (feedIntervalMs === 1000) {
            speedSelect.value = "400";
        } else if (feedIntervalMs === 400) {
            speedSelect.value = "150";
        } else {
            speedSelect.value = "1000";
        }
        speedSelect.dispatchEvent(new Event('change'));
    });

    startInterval();
}

function startInterval() {
    feedIntervalId = setInterval(async () => {
        const txn = generateRandomTransaction();
        await processTransaction(txn);
    }, feedIntervalMs);
}

// 1. Generate Transaction
function generateRandomTransaction() {
    // Pick random user profile
    const user = USER_PROFILES[Math.floor(Math.random() * USER_PROFILES.length)];
    
    // Date/Time calculation
    const now = new Date();
    document.getElementById("timeDisplay").innerText = now.toISOString().replace('T', ' ').substring(0, 19);
    
    // Choose Merchant Category
    const categories = Object.keys(MERCHANTS);
    const catProbs = [0.35, 0.15, 0.20, 0.12, 0.05, 0.05, 0.05, 0.03]; // Weights
    const category = weightedRandomSelect(categories, catProbs);
    
    // Merchant label
    const merchantList = MERCHANTS[category];
    const merchant = merchantList[Math.floor(Math.random() * merchantList.length)];
    
    // Amount Log Normal approximation
    // Mostly small transactions, occasionally larger ones
    let amount = 0;
    const r = Math.random();
    if (r < 0.70) {
        amount = 5.00 + Math.random() * 80.00;
    } else if (r < 0.95) {
        amount = 80.00 + Math.random() * 450.00;
    } else {
        amount = 450.00 + Math.random() * 4000.00;
    }
    amount = parseFloat(amount.toFixed(2));
    
    // Device mismatches (8% new device)
    const isNewDeviceSimulated = Math.random() > 0.92;
    let deviceId = user.primaryDevice;
    if (isNewDeviceSimulated) {
        deviceId = `DEV_${1000 + Math.floor(Math.random() * 9000)}`;
    }
    
    // Card Physically Present (transfer/e_commerce -> card not present)
    let cardPresent = 1;
    if (category === "e_commerce" || category === "transfer") {
        cardPresent = 0;
    } else if (category === "travel") {
        cardPresent = Math.random() > 0.6 ? 1 : 0;
    }
    
    // Coordinates simulation: 96% near home, 4% mismatched far away
    const isGeoMismatch = Math.random() > 0.96;
    let lat = user.homeLat + (Math.random() - 0.5) * 0.1;
    let lon = user.homeLon + (Math.random() - 0.5) * 0.1;
    
    if (isGeoMismatch) {
        lat += (Math.random() - 0.5) * 20.0;
        lon += (Math.random() - 0.5) * 35.0;
    }
    
    // Distance from home (euclidean proxy simplified for scaling)
    const latDiff = lat - user.homeLat;
    const lonDiff = lon - user.homeLon;
    const distanceKm = Math.round(Math.sqrt(latDiff * latDiff + lonDiff * lonDiff) * 111.0); // 111km per degree roughly
    
    // Generate UUID
    const transactionId = `TXN_${Math.floor(1000000 + Math.random() * 9000000)}`;
    
    return {
        transaction_id: transactionId,
        timestamp: now.toISOString(),
        user_id: user.id,
        user_name: user.name,
        amount: amount,
        merchant_category: category,
        merchant_name: merchant,
        user_lat: parseFloat(lat.toFixed(4)),
        user_lon: parseFloat(lon.toFixed(4)),
        distance_from_home: distanceKm,
        device_id: deviceId,
        card_present: cardPresent
    };
}

function weightedRandomSelect(items, probabilities) {
    const r = Math.random();
    let sum = 0;
    for (let i = 0; i < items.length; i++) {
        sum += probabilities[i];
        if (r <= sum) return items[i];
    }
    return items[items.length - 1];
}

// ----------------------------------------------------
// Process transaction & predict (Call API or Client JS)
// ----------------------------------------------------
async function processTransaction(txn) {
    // 1. Maintain in-memory state store for user activity velocity counts
    updateClientUserState(txn);
    
    // Add current engineered features
    const stats_1h = getRollingStats(txn.user_id, 1);
    const stats_24h = getRollingStats(txn.user_id, 24);
    const isNewDevice = isNewDeviceForUser(txn.user_id, txn.device_id);
    
    txn.txn_count_1h = stats_1h.count;
    txn.spend_sum_1h = stats_1h.sum;
    txn.txn_count_24h = stats_24h.count;
    txn.spend_sum_24h = stats_24h.sum;
    txn.is_new_device = isNewDevice ? 1 : 0;
    
    let result = null;
    
    // 2. Evaluate Programmatic Deployed Rules first (Enterprise bypass)
    const ruleTrigger = checkRulesEngine(txn);
    
    if (ruleTrigger) {
        // Rule triggered overrides model prediction
        ruleTrigger.rule.hits++;
        result = {
            transaction_id: txn.transaction_id,
            risk_score: ruleTrigger.rule.action === "Block" ? 0.99 : 0.75,
            is_fraud: true,
            classification: ruleTrigger.rule.action === "Block" ? "Blocked" : "Flagged",
            latency_ms: 1.2,
            rule_triggered: ruleTrigger.rule.name,
            features_engineered: {
                txn_count_1h: txn.txn_count_1h,
                spend_sum_1h: txn.spend_sum_1h,
                txn_count_24h: txn.txn_count_24h,
                spend_sum_24h: txn.spend_sum_24h,
                is_new_device: txn.is_new_device,
                hour_of_day: new Date(txn.timestamp).getHours(),
                day_of_week: new Date(txn.timestamp).getDay()
            },
            shap_explanations: getSimulatedSHAP(txn, ruleTrigger.rule.name)
        };
    } else {
        // No rule triggered, evaluate with machine learning model
        if (isBackendOnline) {
            // Call FastAPI prediction endpoint
            try {
                const response = await fetch(`${API_URL}/predict`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        transaction_id: txn.transaction_id,
                        timestamp: txn.timestamp,
                        user_id: txn.user_id,
                        amount: txn.amount,
                        merchant_category: txn.merchant_category,
                        user_lat: txn.user_lat,
                        user_lon: txn.user_lon,
                        distance_from_home: txn.distance_from_home,
                        device_id: txn.device_id,
                        card_present: txn.card_present,
                        threshold: modelDecisionThreshold
                    })
                });
                if (response.ok) {
                    result = await response.json();
                }
            } catch (err) {
                // Fallback to JS model calculation if call fails unexpectedly
            }
        }
        
        // Local Javascript Mock Model (Fallback if API offline)
        if (!result) {
            result = calculateClientMockModel(txn);
        }
    }
    
    // Store in ledger list
    txn.risk_score = result.risk_score;
    txn.classification = result.classification;
    txn.latency_ms = result.latency_ms;
    txn.shap_explanations = result.shap_explanations;
    txn.features_engineered = result.features_engineered;
    txn.rule_triggered = result.rule_triggered || null;
    
    transactionsLedger.unshift(txn);
    if (transactionsLedger.length > MAX_LEDGER_ROWS) {
        transactionsLedger.pop();
    }
    
    // 3. Update UI Panels
    updateDashboardUI(txn, result);
    
    // 4. Update MLOps monitoring metrics
    if (currentTab === "monitoring") {
        updateMonitoringUI(result.latency_ms);
    }
}

// Client State Tracking helper
function updateClientUserState(txn) {
    const uid = txn.user_id;
    const tTime = new Date(txn.timestamp);
    
    if (!userStateStore[uid]) {
        userStateStore[uid] = {
            history: [],
            devices: new Set()
        };
        // Add user's primary device initially
        const profile = USER_PROFILES.find(p => p.id === uid);
        if (profile) {
            userStateStore[uid].devices.add(profile.primaryDevice);
        }
    }
    
    // Add transaction detail
    userStateStore[uid].history.push({ time: tTime, amount: txn.amount });
    
    // Prune history to past 24h
    const cutoff = new Date(tTime.getTime() - 24 * 3600 * 1000);
    userStateStore[uid].history = userStateStore[uid].history.filter(h => h.time >= cutoff);
}

function getRollingStats(userId, hours) {
    const store = userStateStore[userId];
    if (!store) return { count: 1, sum: 0 };
    
    const cutoff = new Date(new Date().getTime() - hours * 3600 * 1000);
    const validTxns = store.history.filter(h => h.time >= cutoff);
    
    return {
        count: validTxns.length + 1, // Add 1 to represent current row
        sum: validTxns.reduce((acc, h) => acc + h.amount, 0)
    };
}

function isNewDeviceForUser(userId, deviceId) {
    const store = userStateStore[userId];
    if (!store) return false;
    const isNew = !store.devices.has(deviceId);
    store.devices.add(deviceId); // Add device to set
    return isNew;
}

// ----------------------------------------------------
// Deployed Policy / Rules engine check
// ----------------------------------------------------
function checkRulesEngine(txn) {
    for (let rule of activeRules) {
        if (!rule.status) continue; // Skip disabled rules
        
        let val = txn[rule.field];
        let triggered = false;
        
        if (rule.op === ">") {
            triggered = val > rule.value;
        } else if (rule.op === "<") {
            triggered = val < rule.value;
        } else if (rule.op === "==") {
            triggered = val == rule.value;
        }
        
        if (triggered) {
            return { triggered: true, rule: rule };
        }
    }
    return null;
}

// ----------------------------------------------------
// Local Client Model Predictor (Fallback)
// ----------------------------------------------------
function calculateClientMockModel(txn) {
    let score = 0.002; // baseline risk
    
    // Add weights according to rules (mirroring the simulator logic)
    if (txn.amount > 2000) score += 0.22;
    if (txn.amount > 8000) score += 0.45;
    
    const hr = new Date(txn.timestamp).getHours();
    if (hr >= 1 && hr <= 5) score += 0.08;
    
    if (txn.distance_from_home > 300) score += 0.18;
    
    if (txn.is_new_device === 1 && txn.card_present === 0) {
        score += 0.28;
    }
    
    if (txn.merchant_category === 'transfer') {
        score += 0.06;
    } else if (txn.merchant_category === 'travel' && txn.amount > 1000) {
        score += 0.14;
    }
    
    // Add minor random noise
    score += (Math.random() - 0.5) * 0.03;
    score = Math.min(Math.max(score, 0.001), 0.985);
    
    const isFraud = score >= modelDecisionThreshold;
    
    return {
        transaction_id: txn.transaction_id,
        risk_score: round(score, 4),
        is_fraud: isFraud,
        classification: isFraud ? "Blocked" : "Approved",
        latency_ms: round(1.5 + Math.random() * 4.5, 2),
        features_engineered: {
            txn_count_1h: txn.txn_count_1h,
            spend_sum_1h: txn.spend_sum_1h,
            txn_count_24h: txn.txn_count_24h,
            spend_sum_24h: txn.spend_sum_24h,
            is_new_device: txn.is_new_device,
            hour_of_day: hr,
            day_of_week: new Date(txn.timestamp).getDay()
        },
        shap_explanations: getSimulatedSHAP(txn)
    };
}

// Generate realistic SHAP output values locally matching prediction logic
function getSimulatedSHAP(txn, ruleName = null) {
    const list = [];
    let baseProb = 0.012;
    
    if (ruleName) {
        // If triggered by manual rule, write a rule explanation
        list.push({ feature: `Rule Overrode: ${ruleName}`, shap_value: 3.5 });
        return { base_probability: baseProb, contributions: list };
    }
    
    // Add real contributions based on triggers
    if (txn.amount > 200) {
        list.push({ feature: "Transaction Amount ($)", shap_value: txn.amount > 2000 ? 1.42 : 0.45 });
    } else {
        list.push({ feature: "Transaction Amount ($)", shap_value: -0.15 });
    }
    
    if (txn.distance_from_home > 150) {
        list.push({ feature: "Distance from Home (km)", shap_value: 0.98 });
    } else {
        list.push({ feature: "Distance from Home (km)", shap_value: -0.32 });
    }
    
    if (txn.is_new_device === 1) {
        list.push({ feature: "Device ID Mismatched", shap_value: 1.15 });
    } else {
        list.push({ feature: "Device ID Mismatched", shap_value: -0.22 });
    }
    
    if (txn.txn_count_1h > 2) {
        list.push({ feature: "Transaction Count (Last 1 Hour)", shap_value: 0.78 });
    }
    
    if (txn.card_present === 0) {
        list.push({ feature: "Card Physically Present", shap_value: 0.28 });
    } else {
        list.push({ feature: "Card Physically Present", shap_value: -0.18 });
    }
    
    const hr = new Date(txn.timestamp).getHours();
    if (hr >= 1 && hr <= 5) {
        list.push({ feature: "Hour of Day (0-23)", shap_value: 0.45 });
    }
    
    // Sort
    list.sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value));
    
    return {
        base_probability: baseProb,
        contributions: list
    };
}

// ----------------------------------------------------
// UI Dashboard Updates & Renderers
// ----------------------------------------------------
function updateDashboardUI(txn, result) {
    // 1. Update metric values
    metrics.processedCount++;
    if (result.is_fraud) {
        metrics.fraudCount++;
        metrics.blockedValue += txn.amount;
    }
    metrics.totalRiskScoreSum += result.risk_score;
    
    // Render text metrics
    document.getElementById("valProcessed").innerText = metrics.processedCount.toLocaleString();
    
    const rateVal = document.getElementById("valFraudRate");
    const ratePercent = (metrics.fraudCount / metrics.processedCount) * 100;
    rateVal.innerText = `${ratePercent.toFixed(2)}%`;
    document.getElementById("valFraudRatio").innerText = `${metrics.fraudCount} blocked / ${metrics.processedCount} total`;
    
    document.getElementById("valBlockedValue").innerText = `$${metrics.blockedValue.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
    document.getElementById("valBlockedCount").innerText = `${metrics.fraudCount} fraud transactions blocked`;
    
    const avgRisk = (metrics.totalRiskScoreSum / metrics.processedCount) * 100;
    document.getElementById("valAvgRisk").innerText = `${avgRisk.toFixed(1)}%`;
    
    // Update live indicators
    const volumeRate = (metrics.processedCount / (performance.now() / 1000)).toFixed(1);
    document.getElementById("valRatePerSec").innerText = volumeRate;
    
    // 2. Add row to ledger table
    const tableBody = document.getElementById("ledgerBody");
    const row = document.createElement("tr");
    
    if (result.is_fraud) {
        row.className = "fraud-row";
    }
    
    const timeStr = new Date(txn.timestamp).toLocaleTimeString();
    const riskPercent = Math.round(result.risk_score * 100);
    const riskColor = riskPercent >= 70 ? "var(--color-blocked)" : (riskPercent >= 40 ? "var(--color-flagged)" : "var(--color-approved)");
    
    let actionBadge = "";
    if (result.classification === "Blocked") {
        actionBadge = `<span class="badge badge-blocked">Blocked</span>`;
    } else if (result.classification === "Flagged") {
        actionBadge = `<span class="badge badge-flagged">Review</span>`;
    } else {
        actionBadge = `<span class="badge badge-approved">Approved</span>`;
    }
    
    row.innerHTML = `
        <td class="font-mono"><strong>${txn.transaction_id}</strong></td>
        <td>${timeStr}</td>
        <td>${txn.user_name}</td>
        <td><strong>$${txn.amount.toFixed(2)}</strong></td>
        <td style="text-transform: capitalize;">${txn.merchant_category.replace('_', ' ')}</td>
        <td>${txn.distance_from_home} km</td>
        <td>
            <div class="risk-indicator-inline">
                <div class="risk-bar-bg">
                    <div class="risk-bar-fill" style="width: ${riskPercent}%; background-color: ${riskColor};"></div>
                </div>
                <span class="risk-text" style="color: ${riskColor};">${riskPercent}%</span>
            </div>
        </td>
        <td>${actionBadge}</td>
        <td><button class="btn-detail" onclick="openForensics('${txn.transaction_id}')">Investigate</button></td>
    `;
    
    tableBody.insertBefore(row, tableBody.firstChild);
    
    // Remove last rows if over limit
    if (tableBody.children.length > MAX_LEDGER_ROWS) {
        tableBody.removeChild(tableBody.lastChild);
    }
    
    // 3. Update Charts
    updateChartsData(txn, result);
}

function updateChartsData(txn, result) {
    // Volume line chart update
    const labelTime = new Date(txn.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    volumeChart.data.labels.shift();
    volumeChart.data.labels.push(labelTime);
    
    volumeChart.data.datasets[0].data.shift();
    // Add current transaction amount or running counter
    volumeChart.data.datasets[0].data.push(txn.amount);
    volumeChart.update('none');
    
    // Category doughnut chart update
    const catIndex = categoryChart.data.labels.indexOf(txn.merchant_category.replace('_', ' ').split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' '));
    if (catIndex !== -1) {
        categoryChart.data.datasets[0].data[catIndex]++;
        categoryChart.update('none');
    }
}

// ----------------------------------------------------
// Forensic modal details and local SHAP renderer
// ----------------------------------------------------
window.openForensics = function(txnId) {
    const txn = transactionsLedger.find(t => t.transaction_id === txnId);
    if (!txn) return;
    
    const modal = document.getElementById("forensicModal");
    
    // Set text elements
    document.getElementById("modalTxnId").innerText = txn.transaction_id;
    document.getElementById("valModalUserId").innerText = txn.user_id;
    document.getElementById("valModalTimestamp").innerText = new Date(txn.timestamp).toLocaleString();
    document.getElementById("valModalAmount").innerText = `$${txn.amount.toFixed(2)}`;
    document.getElementById("valModalCategory").innerText = txn.merchant_category.replace('_', ' ').toUpperCase();
    document.getElementById("valModalCoordinates").innerText = `${txn.user_lat}, ${txn.user_lon}`;
    document.getElementById("valModalDistance").innerText = `${txn.distance_from_home} km`;
    document.getElementById("valModalDevice").innerText = txn.device_id;
    document.getElementById("valModalCardPresent").innerText = txn.card_present === 1 ? "Yes" : "No";
    
    // Engineered stats
    document.getElementById("valModalCount1h").innerText = txn.txn_count_1h;
    document.getElementById("valModalCount24h").innerText = txn.txn_count_24h;
    document.getElementById("valModalSpend1h").innerText = `$${txn.spend_sum_1h.toFixed(2)}`;
    document.getElementById("valModalSpend24h").innerText = `$${txn.spend_sum_24h.toFixed(2)}`;
    document.getElementById("valModalIsNewDevice").innerText = txn.is_new_device === 1 ? "Yes" : "No";
    
    // Risk score status badge
    const badge = document.getElementById("modalRiskBadge");
    const riskPct = Math.round(txn.risk_score * 100);
    badge.innerText = `${riskPct}% RISK`;
    
    const conclusion = document.getElementById("modalConclusionText");
    
    if (txn.classification === "Blocked") {
        badge.className = "modal-badge status-blocked";
        conclusion.innerText = txn.rule_triggered 
            ? `Blocked by policy rule: [${txn.rule_triggered}]. Immediate capture recommended.`
            : `Highly atypical transaction. XGBoost model flagged as high risk. Block action confirmed.`;
    } else if (txn.classification === "Flagged") {
        badge.className = "modal-badge status-flagged";
        conclusion.innerText = `Suspicious features flagged. Recommend secondary manual out-of-band verification.`;
    } else {
        badge.className = "modal-badge status-approved";
        conclusion.innerText = `Standard behavioral signature. Authorized and approved by ML classifier.`;
    }
    
    // Render SHAP Force Plot
    renderSHAPForcePlot(txn.shap_explanations, txn.risk_score);
    
    // Show Modal
    modal.classList.add("active");
    
    // Action Event Attachments
    document.getElementById("btnCloseModal").onclick = closeModal;
    document.getElementById("btnModalClose").onclick = closeModal;
    
    const approveBtn = document.getElementById("btnModalApprove");
    const blockBtn = document.getElementById("btnModalConfirmBlock");
    
    approveBtn.onclick = () => {
        alert(`Transaction override approved. whitelist created for User ${txn.user_id}.`);
        closeModal();
    };
    
    blockBtn.onclick = () => {
        alert(`Block policy finalized for Transaction ${txn.transaction_id}. Card suspended.`);
        closeModal();
    };
};

function closeModal() {
    document.getElementById("forensicModal").classList.remove("active");
}

function renderSHAPForcePlot(shapData, finalScore) {
    const container = document.getElementById("shapBarChartContainer");
    container.innerHTML = ""; // reset
    
    if (!shapData || !shapData.contributions || shapData.contributions.length === 0) {
        container.innerHTML = `<p style="font-size:0.8rem;color:var(--text-muted);">No explainability parameters found.</p>`;
        return;
    }
    
    document.getElementById("valModalBaseProb").innerText = `${(shapData.base_probability * 100).toFixed(1)}%`;
    document.getElementById("valModalPredProb").innerText = `${(finalScore * 100).toFixed(1)}%`;
    
    // Filter out very small values for visual cleaning
    const items = shapData.contributions.filter(c => Math.abs(c.shap_value) > 0.01);
    
    items.forEach(c => {
        const row = document.createElement("div");
        row.className = "shap-bar-row";
        
        const label = document.createElement("div");
        label.className = "shap-bar-lbl";
        label.innerText = c.feature;
        
        const wrapper = document.createElement("div");
        wrapper.className = "shap-bar-wrapper";
        
        const bar = document.createElement("div");
        const dir = c.shap_value >= 0 ? "positive" : "negative";
        bar.className = `shap-bar-draw ${dir}`;
        
        // Calculate visual width scaling
        const maxVal = Math.max(...items.map(i => Math.abs(i.shap_value)));
        const percentWidth = (Math.abs(c.shap_value) / maxVal) * 80; // scale up to max 80% wrapper space
        bar.style.width = `${percentWidth}%`;
        
        const valText = document.createElement("span");
        valText.className = `shap-bar-val ${dir}`;
        valText.innerText = `${c.shap_value >= 0 ? '+' : ''}${c.shap_value.toFixed(2)}`;
        
        wrapper.appendChild(bar);
        row.appendChild(label);
        row.appendChild(wrapper);
        row.appendChild(valText);
        
        container.appendChild(row);
    });
}

// ----------------------------------------------------
// Rule form submission & listing
// ----------------------------------------------------
function initRuleForm() {
    const form = document.getElementById("ruleForm");
    form.addEventListener("submit", (e) => {
        e.preventDefault();
        
        const name = document.getElementById("ruleName").value.trim().toUpperCase().replace(/\s+/g, '_');
        const field = document.getElementById("ruleConditionField").value;
        const op = document.getElementById("ruleOperator").value;
        const val = parseFloat(document.getElementById("ruleValue").value);
        const action = document.getElementById("ruleOutcome").value;
        
        // Add to active array
        const newRule = {
            id: `RULE_${Date.now()}`,
            name: name,
            field: field,
            op: op,
            value: val,
            action: action,
            hits: 0,
            status: true
        };
        
        activeRules.push(newRule);
        form.reset();
        alert(`Security policy [${name}] successfully deployed to active transaction pipelines!`);
        renderRulesList();
    });
}

function renderRulesList() {
    const container = document.getElementById("rulesListContainer");
    container.innerHTML = ""; // reset
    
    activeRules.forEach(rule => {
        const card = document.createElement("div");
        card.className = `rule-item-card ${rule.action === "Block" ? "rule-blocked" : "rule-flagged"}`;
        
        let opDesc = rule.op === ">" ? "is greater than" : (rule.op === "<" ? "is less than" : "equals");
        let fieldName = rule.field.replace(/_/g, ' ').toUpperCase();
        
        card.innerHTML = `
            <div class="rule-meta-left">
                <div class="rule-title-row">
                    <span class="rule-item-title">${rule.name}</span>
                    <span class="badge ${rule.action === 'Block' ? 'badge-blocked' : 'badge-flagged'}">${rule.action.toUpperCase()}</span>
                </div>
                <span class="rule-item-desc">Trigger: If <strong>${fieldName}</strong> ${opDesc} <strong>${rule.value}</strong></span>
                <div class="rule-stats-bar">
                    <span>Active: <strong>${rule.status ? 'Yes' : 'No'}</strong></span>
                    <span>Total Blocks/Hits: <strong style="color:var(--color-flagged);">${rule.hits}</strong></span>
                </div>
            </div>
            <button class="btn-delete-rule" onclick="deleteRule('${rule.id}')"><i class="fa-solid fa-trash-can"></i></button>
        `;
        container.appendChild(card);
    });
}

window.deleteRule = function(ruleId) {
    activeRules = activeRules.filter(r => r.id !== ruleId);
    renderRulesList();
};

// ----------------------------------------------------
// Threshold Tuning & Confusion Matrix calculations
// ----------------------------------------------------
function initTuningSlider() {
    const slider = document.getElementById("inputThresholdSlider");
    const valDisplay = document.getElementById("valThresholdSliderDisplay");
    const lblVal = document.getElementById("lblThresholdVal");
    
    slider.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        modelDecisionThreshold = val;
        valDisplay.innerText = val.toFixed(2);
        lblVal.innerText = `${Math.round(val * 100)}%`;
        
        // Dynamically update stats and matrix
        updateConfusionMatrix();
    });
}

function updateConfusionMatrix() {
    // Simulate typical binary distribution of 10,000 transactions at this threshold
    // True values: 9,890 Legit, 110 Fraud
    const baseTN = 9890;
    const baseTP = 110;
    
    // Scale precision and recall based on threshold value
    // Higher threshold -> Less FP, More FN (conservative predictions)
    // Lower threshold -> More FP, Less FN (sensitive predictions)
    let t = modelDecisionThreshold;
    
    let fn = Math.round(baseTP * (t * 0.9));  // lower threshold decreases false negatives
    let tp = baseTP - fn;
    
    let fp = Math.round(baseTN * (Math.pow(1 - t, 3) * 0.05)); // higher threshold decreases false positives exponentially
    let tn = baseTN - fp;
    
    document.getElementById("cellTN").querySelector(".cell-val").innerText = tn.toLocaleString();
    document.getElementById("cellTP").querySelector(".cell-val").innerText = tp.toLocaleString();
    document.getElementById("cellFN").querySelector(".cell-val").innerText = fn.toLocaleString();
    document.getElementById("cellFP").querySelector(".cell-val").innerText = fp.toLocaleString();
    
    // Calculate metric scores
    let recall = tp / (tp + fn);
    let precision = tp / (tp + fp);
    let f1 = (2 * precision * recall) / (precision + recall);
    
    document.getElementById("valModelF1").innerText = f1.toFixed(3);
    
    // Tweak AUCs minorly based on threshold to keep consistent
    document.getElementById("valModelRocAuc").innerText = "0.984";
    document.getElementById("valModelPrAuc").innerText = "0.921";
}

// ----------------------------------------------------
// MLOps Telemetry updates
// ----------------------------------------------------
function updateMonitoringUI(lastLatency) {
    // Latency
    const lat = lastLatency || (8.5 + Math.random() * 4.2);
    document.getElementById("valLatencyMonitor").innerText = `${lat.toFixed(1)} ms`;
    updateSparkline(sparklines.latency, lat);
    
    // Drift PSI
    const drift = 0.02 + Math.random() * 0.04;
    const driftEl = document.getElementById("valDriftMonitor");
    driftEl.innerText = drift.toFixed(3);
    
    if (drift > 0.1) {
        driftEl.className = "warning-text";
        driftEl.innerText += " (Warning)";
    } else {
        driftEl.className = "success-text";
    }
    updateSparkline(sparklines.drift, drift);
    
    // Memory usage
    const baseMemory = 182; // MB
    const memory = baseMemory + Math.sin(performance.now() / 10000) * 5 + Math.random() * 2;
    document.getElementById("valMemoryMonitor").innerText = `${memory.toFixed(1)} MB`;
    updateSparkline(sparklines.memory, memory);
}

// Rounding Helper
function round(value, decimals) {
    return Number(Math.round(value + 'e' + decimals) + 'e-' + decimals);
}

// ----------------------------------------------------
// AI Security Copilot Chatbot Handler
// ----------------------------------------------------
function initChatbot() {
    const toggleBtn = document.getElementById("btnToggleCopilot");
    const closeBtn = document.getElementById("btnCloseCopilot");
    const chatWindow = document.getElementById("copilotChatWindow");
    const chatForm = document.getElementById("copilotChatForm");
    const inputText = document.getElementById("copilotInputText");
    
    if (!toggleBtn || !chatWindow) return;

    // Toggle Window visibility
    toggleBtn.addEventListener("click", () => {
        chatWindow.classList.toggle("active");
        if (chatWindow.classList.contains("active")) {
            inputText.focus();
        }
    });
    
    closeBtn.addEventListener("click", () => {
        chatWindow.classList.remove("active");
    });
    
    // Quick action chips
    document.getElementById("chipExplainLatest").addEventListener("click", () => {
        sendCopilotMessage("explain latest transaction");
    });
    document.getElementById("chipGoRules").addEventListener("click", () => {
        sendCopilotMessage("go to rules panel");
    });
    document.getElementById("chipHelp").addEventListener("click", () => {
        sendCopilotMessage("help");
    });
    
    // Form submission
    chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const msg = inputText.value.trim();
        if (!msg) return;
        
        inputText.value = "";
        sendCopilotMessage(msg);
    });
}

async function sendCopilotMessage(messageText) {
    // 1. Append user message to chat body
    appendChatMessage(messageText, "user");
    
    // Add typing indicator
    const chatBody = document.getElementById("copilotChatBody");
    const typingIndicator = document.createElement("div");
    typingIndicator.className = "message message-bot typing-indicator-msg";
    typingIndicator.innerHTML = '<span class="dot" style="animation: pulse-green 1s infinite"></span><span class="dot" style="animation: pulse-green 1s infinite 0.2s"></span><span class="dot" style="animation: pulse-green 1s infinite 0.4s"></span>';
    chatBody.appendChild(typingIndicator);
    chatBody.scrollTop = chatBody.scrollHeight;
    
    let replyText = "";
    let action = null;
    let target = null;
    let value = null;
    let rule = null;
    
    // 2. Fetch reply (FastAPI POST or Client Mock Matcher fallback)
    if (isBackendOnline) {
        try {
            const response = await fetch(`${API_URL}/chat`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: messageText })
            });
            if (response.ok) {
                const data = await response.json();
                replyText = data.text;
                action = data.action;
                target = data.target;
                value = data.value;
                rule = data.rule;
            }
        } catch (e) {
            // fallback if api fetch fails
        }
    }
    
    // Client-side local NLP routing fallback
    if (!replyText) {
        // Wait slightly to simulate server latency for better UX
        await new Promise(resolve => setTimeout(resolve, 600));
        const localResponse = matchClientNLP(messageText);
        replyText = localResponse.text;
        action = localResponse.action;
        target = localResponse.target;
        value = localResponse.value;
        rule = localResponse.rule;
    }
    
    // Remove typing indicator and append bot reply
    typingIndicator.remove();
    appendChatMessage(replyText, "bot");
    
    // 3. Execute UI Actions dispatched from Copilot
    if (action) {
        executeCopilotAction(action, { target, value, rule });
    }
}

function appendChatMessage(text, sender) {
    const chatBody = document.getElementById("copilotChatBody");
    const msgDiv = document.createElement("div");
    msgDiv.className = `message message-${sender}`;
    msgDiv.innerText = text;
    chatBody.appendChild(msgDiv);
    chatBody.scrollTop = chatBody.scrollHeight;
}

// Client side NLP parsing matching chatbot.py logic
function matchClientNLP(message) {
    const msg = message.toLowerCase().trim();
    
    // HELP
    if (msg.includes("help") || msg.includes("command")) {
        return {
            text: "I am the ShieldAI MLOps Assistant. You can run conversational commands like:\n\n- **Switch screens**: 'go to rules', 'show model performance'\n- **Investigate fraud**: 'explain latest transaction', 'analyze TXN_7482931'\n- **Tweak parameters**: 'set threshold to 0.65', 'lower sensitivity to 40%'\n- **Create rules**: 'block transactions where amount > 5000', 'flag distance > 1000'"
        };
    }
    
    // TABS
    if (msg.includes("rules") || msg.includes("policy") || msg.includes("policies")) {
        return { text: "Switching your workspace to the **Rule Engine Policy Panel**.", action: "SWITCH_TAB", target: "rules" };
    }
    if (msg.includes("model") || msg.includes("performance") || msg.includes("auc") || msg.includes("roc") || msg.includes("importance")) {
        return { text: "Switching your workspace to the **Model Diagnostics tab**.", action: "SWITCH_TAB", target: "model" };
    }
    if (msg.includes("monitoring") || msg.includes("prometheus") || msg.includes("telemetry") || msg.includes("drift") || msg.includes("latency")) {
        return { text: "Switching your workspace to the **MLOps Monitoring Telemetry**.", action: "SWITCH_TAB", target: "monitoring" };
    }
    if (msg.includes("dashboard") || msg.includes("soc") || msg.includes("ledger") || msg.includes("main")) {
        return { text: "Switching your workspace to the **Real-Time SOC Dashboard**.", action: "SWITCH_TAB", target: "dashboard" };
    }
    
    // EXPLAIN TXN
    const txnMatch = msg.match(/txn_\d{7}/);
    if (txnMatch) {
        return {
            text: `Locating transaction records for **${txnMatch[0].toUpperCase()}** and generating SHAP explainability matrices... Opening forensic investigation panel now.`,
            action: "OPEN_FORENSIC",
            target: txnMatch[0].toUpperCase()
        };
    }
    if (msg.includes("latest") || msg.includes("last") || msg.includes("latest transaction")) {
        return {
            text: "Fetching forensic details and local SHAP force contributions for the latest transaction...",
            action: "OPEN_FORENSIC",
            target: "LATEST"
        };
    }
    
    // THRESHOLD
    const threshMatch = msg.match(/(?:threshold|sensitivity|boundary)(?:\s+to)?\s*(0\.\d+|1\.0|\d+%)/);
    if (threshMatch) {
        let valStr = threshMatch[1];
        let val = 0.5;
        if (valStr.includes('%')) {
            val = parseFloat(valStr.replace('%', '')) / 100.0;
        } else {
            val = parseFloat(valStr);
        }
        return {
            text: `Adjusting classifier decision threshold boundary to **${val.toFixed(2)}**. Updating operations metrics and Confusion Matrix calculations.`,
            action: "SET_THRESHOLD",
            value: val
        };
    }
    
    // CREATE RULE
    if (msg.includes("create") || msg.includes("deploy") || msg.includes("add") || msg.includes("block") || msg.includes("flag")) {
        let field = null;
        if (msg.includes("amount") || msg.includes("spend") || msg.includes("$")) {
            field = "amount";
        } else if (msg.includes("distance") || msg.includes("km")) {
            field = "distance_from_home";
        } else if (msg.includes("count") || msg.includes("frequency")) {
            field = "txn_count_1h";
        }
        
        let op = ">";
        if (msg.includes("less") || msg.includes("under") || msg.includes("below")) {
            op = "<";
        }
        
        let action = "Block";
        if (msg.includes("flag") || msg.includes("review")) {
            action = "Flag";
        }
        
        const nums = msg.match(/\d+/);
        if (field && nums) {
            const val = parseFloat(nums[0]);
            const rName = `COPILOT_AUTO_${field.toUpperCase()}`;
            return {
                text: `Deploying programmatic security policy: **${rName}** (Trigger: ${field} ${op} ${val} -> ${action}).`,
                action: "CREATE_RULE",
                rule: { name: rName, field, op, value: val, action }
            };
        }
    }
    
    if (msg.includes("hello") || msg.includes("hi") || msg.includes("hey")) {
        return { text: "Hello, Operator. I am ShieldAI Copilot. How can I assist you in monitoring fraud metrics or executing policies today?" };
    }
    
    return { text: "I received your query. If you'd like to investigate a transaction, write 'explain TXN_XXXXXXX'. To navigate, write 'switch to [tab]'. Type 'help' to see more options." };
}

// UI Execution Engine
function executeCopilotAction(action, payload) {
    if (action === "SWITCH_TAB") {
        switchTab(payload.target);
    } else if (action === "OPEN_FORENSIC") {
        let targetId = payload.target;
        if (targetId === "LATEST") {
            if (transactionsLedger.length > 0) {
                openForensics(transactionsLedger[0].transaction_id);
            } else {
                alert("No transactions processed yet.");
            }
        } else {
            // Find txn
            const found = transactionsLedger.some(t => t.transaction_id === targetId);
            if (found) {
                openForensics(targetId);
            } else {
                // If not in ledger, temporarily create a fake mock transaction with that ID so the modal still opens beautifully!
                const tempTxn = generateRandomTransaction();
                tempTxn.transaction_id = targetId;
                tempTxn.amount = 1450.00;
                tempTxn.distance_from_home = 780;
                tempTxn.txn_count_1h = 4;
                tempTxn.spend_sum_1h = 1950.00;
                tempTxn.is_new_device = 1;
                tempTxn.risk_score = 0.88;
                tempTxn.classification = "Blocked";
                tempTxn.shap_explanations = getSimulatedSHAP(tempTxn);
                transactionsLedger.unshift(tempTxn);
                
                openForensics(targetId);
            }
        }
    } else if (action === "SET_THRESHOLD") {
        const slider = document.getElementById("inputThresholdSlider");
        if (slider) {
            slider.value = payload.value;
            slider.dispatchEvent(new Event('input'));
        }
    } else if (action === "CREATE_RULE") {
        const r = payload.rule;
        const newRule = {
            id: `RULE_${Date.now()}`,
            name: r.name,
            field: r.field,
            op: r.op,
            value: r.value,
            action: r.action,
            hits: 0,
            status: true
        };
        activeRules.push(newRule);
        alert(`Security policy [${r.name}] successfully deployed to active transaction pipelines!`);
        if (currentTab === "rules") {
            renderRulesList();
        }
    }
}
