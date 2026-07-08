const AUTH_KEY = "evolytix-auth";
const protectedPages = new Set(["dashboard.html", "history.html", "settings.html"]);
const analysisModels = {
  auditing: {
    label: "Auditing model",
    title: "Upload a transaction CSV.",
    copy:
      "Required columns: Debit, Credit, Amount, Category, Transaction_Type, Payment_Method, Balance, and Previous_Balance.",
    button: "Run Audit Model",
    running: "Running the auditing model...",
    complete: "Audit complete.",
    tableTitle: "First flagged rows",
  },
  forecasting: {
    label: "Financial forecasting model",
    title: "Upload financial data for forecasting.",
    copy:
      "Required columns include Date, Debit, Credit, Amount, Balance, and Category. Tax rows are used when available; otherwise expense rows are used.",
    button: "Run Forecasting Model",
    running: "Running the financial forecasting model...",
    complete: "Forecast complete.",
    tableTitle: "Prediction sample",
  },
  advisory: {
    label: "Optimization advisory model",
    title: "Upload financial data for advisory output.",
    copy:
      "Required columns include Date, Debit, Credit, Amount, Balance, and Category. The model forecasts impact and calculates an optimization target.",
    button: "Run Advisory Model",
    running: "Running the optimization advisory model...",
    complete: "Advisory run complete.",
    tableTitle: "Advisory summary",
  },
};
let selectedAnalysisModel = "auditing";
let preparedRunId = "";

function getCurrentPage() {
  const path = window.location.pathname;
  return path.split("/").pop() || "index.html";
}

function isLoggedIn() {
  return window.localStorage.getItem(AUTH_KEY) === "true";
}

function setLoggedIn(value) {
  window.localStorage.setItem(AUTH_KEY, value ? "true" : "false");
}

function updateNavigation() {
  const loggedIn = isLoggedIn();
  const currentPage = getCurrentPage();

  document.querySelectorAll("[data-auth-nav]").forEach((item) => {
    item.hidden = !loggedIn;
  });

  document.querySelectorAll("[data-login-nav]").forEach((item) => {
    item.hidden = loggedIn;
  });

  document.querySelectorAll("[data-logout-button]").forEach((button) => {
    button.hidden = !loggedIn;
    button.addEventListener("click", () => {
      setLoggedIn(false);
      window.location.href = "index.html";
    });
  });

  document.querySelectorAll(".nav-link").forEach((link) => {
    const href = link.getAttribute("href");
    const isActive = href === currentPage || (href === "index.html" && currentPage === "");
    link.classList.toggle("nav-link-active", isActive);
  });
}

function enforceAccess() {
  const currentPage = getCurrentPage();
  if (protectedPages.has(currentPage) && !isLoggedIn()) {
    window.location.replace("login.html");
    return false;
  }

  if (currentPage === "login.html" && isLoggedIn()) {
    window.location.replace("dashboard.html");
    return false;
  }

  return true;
}

function setupLoginForm() {
  const form = document.querySelector("[data-login-form]");
  if (!form) {
    return;
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    setLoggedIn(true);
    window.location.href = "dashboard.html";
  });
}

function updateAnalysisMode() {
  const chip = document.querySelector("[data-mode-chip]");
  const copy = document.querySelector("[data-mode-copy]");
  const loginPrompt = document.querySelector("[data-mode-login]");
  if (!chip || !copy) {
    return;
  }

  if (isLoggedIn()) {
    chip.textContent = "Adaptive Mode Active";
    chip.classList.add("mode-chip-adaptive");
    copy.textContent =
      "Company tools are unlocked, and your adaptive workspace is available.";
    if (loginPrompt) {
      loginPrompt.hidden = true;
    }
    return;
  }

  chip.textContent = "Static Mode Active";
  chip.classList.remove("mode-chip-adaptive");
  copy.textContent =
    "You are using the public experience. Log in anytime to unlock dashboard, history, and settings.";
  if (loginPrompt) {
    loginPrompt.hidden = false;
  }
}

function formatPercent(value) {
  return `${(Number(value || 0) * 100).toFixed(2)}%`;
}

function formatMetricValue(metric) {
  const value = metric.value;
  if (typeof value !== "number") {
    return String(value ?? "");
  }

  if (metric.format === "percent") {
    return formatPercent(value);
  }
  if (metric.format === "currency") {
    return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
  }
  if (metric.format === "decimal") {
    return value.toFixed(3);
  }
  return Number.isInteger(value) ? String(value) : value.toFixed(2);
}

function setAuditStatus(message, tone = "muted") {
  const status = document.querySelector("[data-audit-status]");
  if (!status) {
    return;
  }

  status.textContent = message;
  status.dataset.tone = tone;
}

function renderFlaggedRows(rows) {
  const wrap = document.querySelector("[data-flagged-wrap]");
  const head = document.querySelector("[data-flagged-head]");
  const body = document.querySelector("[data-flagged-body]");
  if (!wrap || !head || !body) {
    return;
  }

  head.innerHTML = "";
  body.innerHTML = "";

  if (!rows.length) {
    wrap.hidden = true;
    return;
  }

  const columns = Object.keys(rows[0]).slice(0, 8);
  const headerRow = document.createElement("tr");
  columns.forEach((column) => {
    const cell = document.createElement("th");
    cell.textContent = column;
    headerRow.appendChild(cell);
  });
  head.appendChild(headerRow);

  rows.forEach((row) => {
    const tableRow = document.createElement("tr");
    columns.forEach((column) => {
      const cell = document.createElement("td");
      const value = row[column];
      cell.textContent = typeof value === "number" ? value.toFixed(4) : String(value ?? "");
      tableRow.appendChild(cell);
    });
    body.appendChild(tableRow);
  });

  wrap.hidden = false;
}

function getDefaultMetrics(result) {
  const summary = result.summary || {};
  const adaptiveLayer = result.adaptiveLayer || {};
  const events = adaptiveLayer.events || [];
  return [
    { label: "Rows", value: summary.totalRows ?? 0 },
    { label: "Anomalies", value: summary.finalAnomalyCount ?? 0 },
    { label: "Rate", value: summary.anomalyRate ?? 0, format: "percent" },
    { label: "Adaptive events", value: summary.adaptationEvents ?? events.length },
  ];
}

function renderMetrics(result) {
  const results = document.querySelector("[data-audit-results]");
  if (!results) {
    return;
  }

  const metrics = result.metrics || getDefaultMetrics(result);
  results.innerHTML = "";
  metrics.slice(0, 5).forEach((metric) => {
    const card = document.createElement("article");
    card.className = "info-card metric-card";

    const label = document.createElement("span");
    label.className = "small-label";
    label.textContent = metric.label;

    const value = document.createElement("strong");
    value.textContent = formatMetricValue(metric);

    card.append(label, value);
    results.appendChild(card);
  });
  results.hidden = false;
}

function renderAuditResults(result) {
  const results = document.querySelector("[data-audit-results]");
  const plotPanel = document.querySelector("[data-plot-panel]");
  const plot = document.querySelector("[data-result-plot]");
  const tableTitle = document.querySelector("[data-result-table-title]");
  const modelOutput = document.querySelector("[data-model-output]");
  if (!results) {
    return;
  }

  if (modelOutput) {
    modelOutput.hidden = false;
  }
  renderMetrics(result);
  if (tableTitle) {
    tableTitle.textContent =
      result.tableTitle || analysisModels[selectedAnalysisModel].tableTitle;
  }
  renderFlaggedRows(result.flaggedRows || []);
  if (plotPanel && plot && result.plotUrl) {
    plot.src = `${result.plotUrl}?t=${Date.now()}`;
    plotPanel.hidden = false;
  } else if (plotPanel && plot) {
    plot.removeAttribute("src");
    plotPanel.hidden = true;
  }

}

function updateSelectedAnalysisModel(model) {
  selectedAnalysisModel = model;
  const config = analysisModels[model];
  const label = document.querySelector("[data-workspace-label]");
  const title = document.querySelector("[data-workspace-title]");
  const copy = document.querySelector("[data-workspace-copy]");
  const submit = document.querySelector("[data-audit-form] button[type='submit']");
  const results = document.querySelector("[data-audit-results]");
  const flaggedWrap = document.querySelector("[data-flagged-wrap]");
  const plotPanel = document.querySelector("[data-plot-panel]");
  const modelOutput = document.querySelector("[data-model-output]");

  document.querySelectorAll("[data-service-card]").forEach((card) => {
    card.classList.toggle("service-card-active", card.dataset.serviceCard === model);
  });

  if (label) label.textContent = config.label;
  if (title) title.textContent = config.title;
  if (copy) copy.textContent = config.copy;
  if (submit) submit.textContent = config.button;
  if (results) results.hidden = true;
  if (flaggedWrap) flaggedWrap.hidden = true;
  if (plotPanel) plotPanel.hidden = true;
  if (modelOutput) modelOutput.hidden = true;
  setAuditStatus("");
}

function setupAnalysisModelChoices() {
  document.querySelectorAll("[data-service-choice]").forEach((button) => {
    button.addEventListener("click", () => {
      updateSelectedAnalysisModel(button.dataset.serviceChoice);
    });
  });
}

function renderPreprocessingOutput(result) {
  const output = document.querySelector("[data-preprocess-output]");
  const modelSelection = document.querySelector("[data-model-selection]");
  const summary = document.querySelector("[data-preprocess-summary]");
  const cleanedCsvLink = document.querySelector("[data-cleaned-csv-link]");
  const qualityReportLink = document.querySelector("[data-quality-report-link]");
  const preprocessing = result.preprocessing || {};

  preparedRunId = result.runId || "";
  if (cleanedCsvLink) {
    cleanedCsvLink.href = preprocessing.cleanedCsvUrl || "#";
  }
  if (qualityReportLink) {
    qualityReportLink.href = preprocessing.qualityReportUrl || "#";
  }
  if (summary) {
    summary.textContent = `Prepared ${preprocessing.preparedParquet || "model input"} from the uploaded file.`;
  }
  if (output) {
    output.hidden = false;
  }
  if (modelSelection) {
    modelSelection.hidden = false;
  }
}

function resetModelOutput() {
  const results = document.querySelector("[data-audit-results]");
  const flaggedWrap = document.querySelector("[data-flagged-wrap]");
  const plotPanel = document.querySelector("[data-plot-panel]");
  const modelOutput = document.querySelector("[data-model-output]");

  if (results) results.hidden = true;
  if (flaggedWrap) flaggedWrap.hidden = true;
  if (plotPanel) plotPanel.hidden = true;
  if (modelOutput) modelOutput.hidden = true;
}

function setupPreprocessForm() {
  const form = document.querySelector("[data-preprocess-form]");
  const fileInput = document.querySelector("[data-audit-file]");
  if (!form || !fileInput) {
    return;
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const file = fileInput.files[0];
    if (!file) {
      setAuditStatus("Choose a CSV file first.", "error");
      return;
    }

    const submitButton = form.querySelector("button[type='submit']");
    submitButton.disabled = true;
    preparedRunId = "";
    resetModelOutput();
    const preprocessOutput = document.querySelector("[data-preprocess-output]");
    const modelSelection = document.querySelector("[data-model-selection]");
    if (preprocessOutput) preprocessOutput.hidden = true;
    if (modelSelection) modelSelection.hidden = true;
    setAuditStatus("Preprocessing the uploaded file...", "muted");

    try {
      const csvText = await file.text();
      const response = await fetch("/api/preprocess", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ csvText }),
      });
      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || "Preprocessing failed.");
      }

      renderPreprocessingOutput(result);
      setAuditStatus("Preprocessing complete. Select a model to continue.", "success");
    } catch (error) {
      setAuditStatus(error.message, "error");
    } finally {
      submitButton.disabled = false;
    }
  });
}

function setupRunModelButton() {
  const button = document.querySelector("[data-run-model-button]");
  if (!button) {
    return;
  }

  button.addEventListener("click", async () => {
    if (!preparedRunId) {
      setAuditStatus("Preprocess a file before running a model.", "error");
      return;
    }

    button.disabled = true;
    resetModelOutput();
    setAuditStatus(analysisModels[selectedAnalysisModel].running, "muted");

    try {
      const response = await fetch("/api/model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: selectedAnalysisModel, runId: preparedRunId }),
      });
      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || "The model request failed.");
      }

      renderAuditResults(result);
      setAuditStatus(analysisModels[selectedAnalysisModel].complete, "success");
    } catch (error) {
      setAuditStatus(error.message, "error");
    } finally {
      button.disabled = false;
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  if (!enforceAccess()) {
    return;
  }

  updateNavigation();
  setupLoginForm();
  updateAnalysisMode();
  setupAnalysisModelChoices();
  updateSelectedAnalysisModel(selectedAnalysisModel);
  setupPreprocessForm();
  setupRunModelButton();
});
