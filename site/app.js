async function fetchText(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${path}`);
  }
  return response.text();
}

async function fetchJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${path}`);
  }
  return response.json();
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];

    if (char === '"') {
      if (inQuotes && next === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (char === "," && !inQuotes) {
      row.push(current);
      current = "";
      continue;
    }

    if ((char === "\n" || char === "\r") && !inQuotes) {
      if (char === "\r" && next === "\n") {
        i += 1;
      }
      row.push(current);
      current = "";
      if (row.some(value => value !== "")) {
        rows.push(row);
      }
      row = [];
      continue;
    }

    current += char;
  }

  if (current.length > 0 || row.length > 0) {
    row.push(current);
    rows.push(row);
  }

  const [header, ...body] = rows;
  return body.map(entry => Object.fromEntries(header.map((key, index) => [key, entry[index] ?? ""])));
}

function formatMetric(value) {
  return Number(value).toFixed(2);
}

function updateHero(metrics) {
  document.getElementById("best-model").textContent = metrics.best_model;
  document.getElementById("precision").textContent = formatMetric(metrics.test_metrics.precision);
  document.getElementById("recall").textContent = formatMetric(metrics.test_metrics.recall);
  document.getElementById("f1-score").textContent = formatMetric(metrics.test_metrics.f1_score);
  document.getElementById("roc-auc").textContent = formatMetric(metrics.test_metrics.roc_auc);
}

function populateLeaderboard(rows) {
  const tbody = document.querySelector("#leaderboard-table tbody");
  tbody.innerHTML = "";

  rows.forEach(row => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.model_name}</td>
      <td>${formatMetric(row.precision)}</td>
      <td>${formatMetric(row.recall)}</td>
      <td>${formatMetric(row.f1_score)}</td>
      <td>${formatMetric(row.roc_auc)}</td>
    `;
    tbody.appendChild(tr);
  });
}

function populatePredictions(rows) {
  const tbody = document.querySelector("#predictions-table tbody");
  tbody.innerHTML = "";

  rows.slice(0, 8).forEach(row => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.timestamp}</td>
      <td>${row.country_code}</td>
      <td>${row.login_attempts_last_1h}</td>
      <td>${row.failed_logins_last_24h}</td>
      <td>${row.payload_anomaly_score}</td>
      <td>${Number(row.fraud_probability).toFixed(4)}</td>
      <td><span class="badge">${row.prediction === "1" ? "Fraud" : "Legitimate"}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

async function init() {
  try {
    const [metrics, leaderboardCsv, predictionsCsv] = await Promise.all([
      fetchJson("../artifacts/metrics.json"),
      fetchText("../artifacts/model_leaderboard.csv"),
      fetchText("../artifacts/sample_predictions.csv")
    ]);

    updateHero(metrics);
    populateLeaderboard(parseCsv(leaderboardCsv));
    populatePredictions(parseCsv(predictionsCsv));
  } catch (error) {
    console.error(error);
    document.getElementById("best-model").textContent = "Unable to load";
  }
}

init();
