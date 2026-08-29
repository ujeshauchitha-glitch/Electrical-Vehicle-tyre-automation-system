/**
 * EV Tyre Intelligence — Frontend Controller
 *
 * Collects slider values, POSTs to the FastAPI backend,
 * and renders the estimation results.
 */

const API_BASE = "http://localhost:8000";

// ---------------------------------------------------------------------------
// Slider value sync
// ---------------------------------------------------------------------------

const sliders = {
    pfl:  { el: "pfl",  val: "val-pfl",  fmt: v => v },
    pfr:  { el: "pfr",  val: "val-pfr",  fmt: v => v },
    prl:  { el: "prl",  val: "val-prl",  fmt: v => v },
    prr:  { el: "prr",  val: "val-prr",  fmt: v => v },
    rf:   { el: "rf",   val: "val-rf",   fmt: v => parseFloat(v).toFixed(3) },
    rr:   { el: "rr",   val: "val-rr",   fmt: v => parseFloat(v).toFixed(3) },
    temp: { el: "temp", val: "val-temp", fmt: v => v },
};

Object.values(sliders).forEach(s => {
    const input = document.getElementById(s.el);
    const display = document.getElementById(s.val);
    input.addEventListener("input", () => {
        display.textContent = s.fmt(input.value);
    });
});

function getInput() {
    return {
        pressure_fl: parseFloat(document.getElementById("pfl").value),
        pressure_fr: parseFloat(document.getElementById("pfr").value),
        pressure_rl: parseFloat(document.getElementById("prl").value),
        pressure_rr: parseFloat(document.getElementById("prr").value),
        ratio_front: parseFloat(document.getElementById("rf").value),
        ratio_rear:  parseFloat(document.getElementById("rr").value),
        temperature: parseFloat(document.getElementById("temp").value),
    };
}

// ---------------------------------------------------------------------------
// API calls
// ---------------------------------------------------------------------------

async function runEstimator() {
    const status = document.getElementById("status-msg");
    const results = document.getElementById("results");

    status.textContent = "Running estimator...";
    status.className = "status-msg loading";
    results.classList.add("hidden");

    try {
        const resp = await fetch(`${API_BASE}/api/estimate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(getInput()),
        });

        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        renderResults(data);
        status.textContent = `Converged in ${data.iterations} iterations`;
        status.className = "status-msg success";
    } catch (err) {
        status.textContent = `Error: ${err.message}. Is the backend running?`;
        status.className = "status-msg error";
    }
}

async function runSimulation() {
    const status = document.getElementById("status-msg");
    const results = document.getElementById("results");

    status.textContent = "Running simulation...";
    status.className = "status-msg loading";
    results.classList.add("hidden");

    try {
        const resp = await fetch(`${API_BASE}/api/simulation`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ seed: null, iters: 6 }),
        });

        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        renderResults(data.estimate);
        renderTruth(data.truth);
        status.textContent = `Simulation done — ${data.estimate.iterations} iterations`;
        status.className = "status-msg success";
    } catch (err) {
        status.textContent = `Error: ${err.message}. Is the backend running?`;
        status.className = "status-msg error";
    }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function renderResults(est) {
    const results = document.getElementById("results");
    results.classList.remove("hidden");

    // Tread depth
    const tread = est.states.filter(s => s.name.startsWith("tread_"));
    document.getElementById("tread-table").innerHTML = tread.map(s => {
        const corner = s.name.split("_")[1];
        const barW = Math.max(5, (s.value / 8) * 100);
        const color = s.value < 1.6 ? "#e74c3c" : s.value < 3.0 ? "#f39c12" : "#27ae60";
        return `<div class="data-row">
            <span class="corner">${corner}</span>
            <div class="bar-container"><div class="bar" style="width:${barW}%;background:${color}"></div></div>
            <span class="value">${s.value.toFixed(2)} mm</span>
            <span class="sigma">&plusmn; ${s.sigma.toFixed(3)}</span>
        </div>`;
    }).join("");

    // Pressure (cold-equivalent)
    const cold = est.cold_pressure;
    document.getElementById("pressure-table").innerHTML = ["FL","FR","RL","RR"].map(c => {
        const val = cold[c];
        const flag = val < 220 ? " ⚠ LOW" : "";
        return `<div class="data-row">
            <span class="corner">${c}</span>
            <span class="value">${val.toFixed(0)} kPa${flag}</span>
        </div>`;
    }).join("");

    // Alignment
    const toe = est.states.find(s => s.name === "toe^2");
    const camber = est.states.find(s => s.name === "camber");
    const toeMag = toe ? Math.sqrt(Math.max(0, toe.value)) : 0;
    document.getElementById("alignment-table").innerHTML = `
        <div class="data-row">
            <span>|toe|</span>
            <span class="value">${toeMag.toFixed(2)}&deg;</span>
            <span class="sigma">&plusmn; (magnitude only)</span>
        </div>
        <div class="data-row">
            <span>camber</span>
            <span class="value">${camber ? camber.value.toFixed(2) : "—"}&deg;</span>
            <span class="sigma">${camber && camber.observability === "NO_INFORMATION" ? "unobservable" : ""}</span>
        </div>`;

    // Torque ceiling
    const tc = est.torque_ceiling;
    document.getElementById("torque-info").innerHTML = `
        <div class="data-row"><span>Wet</span><span class="value">${tc.wet_Nm.toFixed(0)} N.m</span></div>
        <div class="data-row"><span>Dry</span><span class="value">${tc.dry_Nm.toFixed(0)} N.m</span></div>`;

    // Energy
    document.getElementById("energy-info").innerHTML = `
        <div class="data-row"><span>Excess road load</span>
        <span class="value ${est.recoverable_energy_pct > 0 ? 'warn' : ''}">${est.recoverable_energy_pct > 0 ? '+' : ''}${est.recoverable_energy_pct.toFixed(1)}%</span></div>`;

    // Variance reduction bars
    document.getElementById("variance-bars").innerHTML = est.states
        .filter(s => !s.name.startsWith("press_"))
        .map(s => {
            const pct = Math.min(100, Math.max(0, (s.variance_reduction - 1) / 20 * 100));
            const color = s.observability === "NO_INFORMATION" ? "#95a5a6" : "#3498db";
            return `<div class="data-row">
                <span class="corner">${s.name}</span>
                <div class="bar-container"><div class="bar" style="width:${pct}%;background:${color}"></div></div>
                <span class="value">x${s.variance_reduction.toFixed(1)}</span>
            </div>`;
        }).join("");
}

function renderTruth(truth) {
    // Prepend truth info when simulating
    const tread = document.getElementById("tread-table");
    const header = `<div class="truth-header">Ground truth shown in italic</div>`;
    tread.innerHTML = header + tread.innerHTML;
}

// ---------------------------------------------------------------------------
// Event listeners
// ---------------------------------------------------------------------------

document.getElementById("run-btn").addEventListener("click", runEstimator);
document.getElementById("sim-btn").addEventListener("click", runSimulation);

// Check backend health on load
fetch(`${API_BASE}/api/health`)
    .then(r => r.json())
    .then(d => {
        document.getElementById("status-msg").textContent =
            `Connected (v${d.version}). Adjust inputs and click Run.`;
    })
    .catch(() => {
        document.getElementById("status-msg").textContent =
            "Backend not reachable. Start with: uvicorn backend.main:app --reload";
        document.getElementById("status-msg").className = "status-msg error";
    });
