import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";

const tabs = [
  { id: "ops", label: "Operations" },
  { id: "model", label: "Model & Predict" },
  { id: "quality", label: "Data Quality" },
  { id: "docs", label: "Docs" }
];

const STORAGE_KEYS = {
  activeTab: "market_lens_active_tab",
  ticker: "market_lens_ticker",
  autoRefresh: "market_lens_auto_refresh",
  refreshMs: "market_lens_refresh_ms"
};

function fmtTime(value) {
  if (!value) return "never";
  try {
    return new Date(value).toLocaleString();
  } catch {
    return String(value);
  }
}

function JsonBlock({ data, label = "data" }) {
  const copyJson = async () => {
    try {
      await navigator.clipboard.writeText(JSON.stringify(data, null, 2));
    } catch {
      // no-op
    }
  };

  return (
    <div>
      <div className="mb-2 flex items-center justify-between text-xs text-mute">
        <span>{label}</span>
        <button
          className="rounded border border-line px-2 py-1 hover:border-accent/70 hover:text-ink"
          onClick={copyJson}
          title={`Copy ${label} JSON`}
        >
          Copy
        </button>
      </div>
      <pre className="overflow-auto rounded-lg border border-line/70 bg-bg/70 p-3 text-xs text-mute">
        {JSON.stringify(data, null, 2)}
      </pre>
    </div>
  );
}

function StatCard({ label, value, tone = "ink" }) {
  const toneClass = {
    ink: "text-ink",
    good: "text-good",
    warn: "text-warn",
    bad: "text-bad",
    accent: "text-accent"
  }[tone] || "text-ink";
  return (
    <div className="rounded-xl border border-line bg-panel/70 p-4 shadow-glow">
      <div className="text-xs uppercase tracking-wide text-mute">{label}</div>
      <div className={`mt-2 text-2xl font-display font-semibold ${toneClass}`}>{value}</div>
    </div>
  );
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json();
}

export default function App() {
  const [activeTab, setActiveTab] = useState(localStorage.getItem(STORAGE_KEYS.activeTab) || "ops");
  const [ticker, setTicker] = useState(localStorage.getItem(STORAGE_KEYS.ticker) || "SPY");
  const [autoRefresh, setAutoRefresh] = useState(
    (localStorage.getItem(STORAGE_KEYS.autoRefresh) || "true") === "true"
  );
  const [refreshMs, setRefreshMs] = useState(Number(localStorage.getItem(STORAGE_KEYS.refreshMs) || 15000));
  const [logTickerFilter, setLogTickerFilter] = useState("");
  const [logPredFilter, setLogPredFilter] = useState("all");
  const [busy, setBusy] = useState({});
  const [messages, setMessages] = useState([]);
  const [lastError, setLastError] = useState("");
  const [lastUpdated, setLastUpdated] = useState({});

  const [health, setHealth] = useState(null);
  const [dataStatus, setDataStatus] = useState(null);
  const [quality, setQuality] = useState(null);
  const [ingestStatus, setIngestStatus] = useState(null);
  const [pipelineStatus, setPipelineStatus] = useState(null);
  const [modelStatus, setModelStatus] = useState(null);
  const [modelHistory, setModelHistory] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [predictionLogs, setPredictionLogs] = useState([]);
  const [docs, setDocs] = useState({ readme_markdown: "", project_overview_text: "" });

  const pushMessage = (text, tone = "info") => {
    setMessages((prev) => [{ id: Date.now(), text, tone }, ...prev].slice(0, 8));
  };

  const runAction = async (key, fn) => {
    setBusy((prev) => ({ ...prev, [key]: true }));
    try {
      const data = await fn();
      setLastError("");
      pushMessage(`${key} completed`, "ok");
      return data;
    } catch (err) {
      setLastError(`${key}: ${err.message}`);
      pushMessage(`${key} failed: ${err.message}`, "err");
      throw err;
    } finally {
      setBusy((prev) => ({ ...prev, [key]: false }));
    }
  };

  const refreshAll = async () => {
    const now = new Date().toISOString();
    await Promise.all([
      api("/health").then((d) => {
        setHealth(d);
        setLastUpdated((prev) => ({ ...prev, health: now }));
      }),
      api("/data/status").then((d) => {
        setDataStatus(d);
        setLastUpdated((prev) => ({ ...prev, dataStatus: now }));
      }),
      api("/data/quality").then((d) => {
        setQuality(d);
        setLastUpdated((prev) => ({ ...prev, quality: now }));
      }),
      api("/ingest/status").then((d) => {
        setIngestStatus(d);
        setLastUpdated((prev) => ({ ...prev, ingestStatus: now }));
      }),
      api("/pipeline/status").then((d) => {
        setPipelineStatus(d);
        setLastUpdated((prev) => ({ ...prev, pipelineStatus: now }));
      }),
      api("/model/status").then((d) => {
        setModelStatus(d);
        setLastUpdated((prev) => ({ ...prev, modelStatus: now }));
      }),
      api("/model/history?limit=20").then((d) => {
        setModelHistory(d.rows || []);
        setLastUpdated((prev) => ({ ...prev, modelHistory: now }));
      }),
      api("/prediction/logs?limit=100").then((d) => {
        setPredictionLogs(d.rows || []);
        setLastUpdated((prev) => ({ ...prev, predictionLogs: now }));
      }),
      api("/docs/text").then((d) => {
        setDocs(d);
        setLastUpdated((prev) => ({ ...prev, docs: now }));
      })
    ]);
  };

  useEffect(() => {
    refreshAll().catch((err) => {
      setLastError(`Initial load: ${err.message}`);
      pushMessage(`Initial load failed: ${err.message}`, "err");
    });
  }, []);

  useEffect(() => {
    if (!autoRefresh) return () => null;
    const t = setInterval(() => {
      refreshAll().catch((err) => {
        setLastError(`refresh: ${err.message}`);
        pushMessage(`refresh failed: ${err.message}`, "err");
      });
    }, refreshMs);
    return () => clearInterval(t);
  }, [autoRefresh, refreshMs]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.activeTab, activeTab);
  }, [activeTab]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.ticker, ticker);
  }, [ticker]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.autoRefresh, String(autoRefresh));
  }, [autoRefresh]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEYS.refreshMs, String(refreshMs));
  }, [refreshMs]);

  const totalRows = useMemo(() => {
    if (!dataStatus?.totals) return 0;
    return Object.values(dataStatus.totals).reduce((a, b) => a + Number(b || 0), 0);
  }, [dataStatus]);

  const kpis = useMemo(() => {
    const dups = quality?.duplicate_keys || {};
    const nulls = quality?.null_checks || {};
    const fresh = quality?.freshness || {};
    return {
      duplicateCount:
        Number(dups.news_source_url_duplicates || 0) + Number(dups.market_ticker_timestamp_duplicates || 0),
      nullRatio: Number(nulls.news_published_at_null_ratio || 0),
      newsStale: !!fresh.news_stale,
      marketStale: !!fresh.market_stale
    };
  }, [quality]);

  const filteredLogs = useMemo(() => {
    return predictionLogs.filter((row) => {
      const tickerPass = !logTickerFilter || row.ticker?.toUpperCase().includes(logTickerFilter.toUpperCase());
      const predPass = logPredFilter === "all" || row.prediction === logPredFilter;
      return tickerPass && predPass;
    });
  }, [predictionLogs, logTickerFilter, logPredFilter]);

  const latestModelRun = modelHistory[0] || null;
  const previousModelRun = modelHistory[1] || null;
  const metricDelta = (current, previous) => {
    if (current == null || previous == null) return null;
    return Number(current) - Number(previous);
  };

  const latestRunStatus = pipelineStatus?.latest_run?.status || ingestStatus?.latest_run?.status || "none";
  const statusTone =
    latestRunStatus === "success" ? "good" : latestRunStatus === "failed" ? "bad" : latestRunStatus === "running" ? "warn" : "ink";

  return (
    <div className="min-h-screen bg-[radial-gradient(circle_at_10%_10%,#1e2f5f_0,#0b1020_45%,#050810_100%)] font-body text-ink">
      <div className="mx-auto max-w-7xl px-4 py-6 md:px-8 md:py-10">
        <header className="mb-6 rounded-2xl border border-line bg-panel/70 p-5 shadow-glow">
          <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
            <div>
              <h1 className="font-display text-3xl font-semibold">Market Lens Control Deck</h1>
              <p className="mt-1 text-sm text-mute">
                Operate the price-first pipeline, model training, prediction, and quality checks from one screen.
              </p>
            </div>
            <button
              className="rounded-lg border border-accent/60 bg-accent/20 px-4 py-2 text-sm font-medium hover:bg-accent/30"
              onClick={() => runAction("refresh", refreshAll)}
              title="Refresh all dashboard panels (GET /health, /data/*, /ingest/status, /pipeline/status, /model/status, /prediction/logs, /docs/text)"
            >
              {busy.refresh ? "Refreshing..." : "Refresh Now"}
            </button>
          </div>
          <div className="mt-4 flex flex-wrap items-center gap-3 text-xs text-mute">
            <label className="inline-flex items-center gap-2">
              <input type="checkbox" checked={autoRefresh} onChange={(e) => setAutoRefresh(e.target.checked)} />
              Auto refresh
            </label>
            <label className="inline-flex items-center gap-2">
              Interval
              <select
                className="rounded border border-line bg-bg px-2 py-1 text-ink"
                value={refreshMs}
                onChange={(e) => setRefreshMs(Number(e.target.value))}
              >
                <option value={5000}>5s</option>
                <option value={15000}>15s</option>
                <option value={30000}>30s</option>
              </select>
            </label>
            <span>Last refresh: {fmtTime(lastUpdated.health)}</span>
          </div>
          {lastError && (
            <div className="mt-3 rounded-lg border border-bad/60 bg-bad/10 px-3 py-2 text-xs text-bad">
              Last error: {lastError}
            </div>
          )}
        </header>

        <section className="mb-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard label="Health" value={health?.status || "unknown"} tone={health?.status === "ok" ? "good" : "warn"} />
          <StatCard label="Total Rows" value={totalRows} />
          <StatCard label="Latest Model" value={modelStatus?.metadata?.version_id || "none"} tone="accent" />
          <StatCard label="Latest Run" value={latestRunStatus} tone={statusTone} />
        </section>

        <section className="mb-6 flex flex-wrap gap-2">
          <span className={`rounded-full border px-3 py-1 text-xs ${kpis.newsStale ? "border-bad text-bad" : "border-good text-good"}`}>
            news_stale: {String(kpis.newsStale)}
          </span>
          <span className={`rounded-full border px-3 py-1 text-xs ${kpis.marketStale ? "border-bad text-bad" : "border-good text-good"}`}>
            market_stale: {String(kpis.marketStale)}
          </span>
          <span className="rounded-full border border-line px-3 py-1 text-xs text-mute">duplicates: {kpis.duplicateCount}</span>
          <span className="rounded-full border border-line px-3 py-1 text-xs text-mute">
            null_ratio: {(kpis.nullRatio * 100).toFixed(2)}%
          </span>
        </section>

        <nav className="mb-5 flex flex-wrap gap-2">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              className={`rounded-lg border px-4 py-2 text-sm ${
                activeTab === tab.id
                  ? "border-accent bg-accent/20 text-ink"
                  : "border-line bg-panel/50 text-mute hover:text-ink"
              }`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        {activeTab === "ops" && (
          <section className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
            <div className="rounded-2xl border border-line bg-panel/70 p-5">
              <h2 className="font-display text-xl">Pipeline Actions</h2>
              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                <button
                  className="rounded-lg border border-line bg-bg/60 px-4 py-3 text-left hover:border-accent/70"
                  onClick={() =>
                    runAction("ingest_run", async () => {
                      const out = await api("/ingest/run", { method: "POST" });
                      await refreshAll();
                      return out;
                    })
                  }
                  title="POST /ingest/run"
                >
                  <div className="font-medium">Run Market Ingestion</div>
                  <div className="text-xs text-mute">Market data pull (news path paused)</div>
                </button>
                <button
                  className="rounded-lg border border-line bg-bg/60 px-4 py-3 text-left hover:border-accent/70"
                  onClick={() =>
                    runAction("pipeline_run", async () => {
                      const out = await api("/pipeline/run", { method: "POST" });
                      await refreshAll();
                      return out;
                    })
                  }
                  title="POST /pipeline/run"
                >
                  <div className="font-medium">Run Features + Labels</div>
                  <div className="text-xs text-mute">Price snapshot + label jobs</div>
                </button>
                <button
                  className="rounded-lg border border-line bg-bg/60 px-4 py-3 text-left hover:border-accent/70"
                  onClick={() => {
                    if (!window.confirm("Train model now? This overwrites latest model pointers.")) return;
                    return runAction("model_train", async () => {
                      const out = await api("/model/train", { method: "POST" });
                      await refreshAll();
                      return out;
                    });
                  }}
                  title="POST /model/train"
                >
                  <div className="font-medium">Train Model</div>
                  <div className="text-xs text-mute">Versioned artifact save</div>
                </button>
                <button
                  className="rounded-lg border border-accent/70 bg-accent/20 px-4 py-3 text-left hover:bg-accent/30"
                  onClick={() => {
                    if (!window.confirm("Run full refresh? This runs ingestion, pipeline, and model training.")) return;
                    return runAction("run_full", async () => {
                      const out = await api("/run/full?train_model=true", { method: "POST" });
                      await refreshAll();
                      return out;
                    });
                  }}
                  title="POST /run/full?train_model=true"
                >
                  <div className="font-medium">Run Full Refresh</div>
                  <div className="text-xs text-mute">Market ingest {"->"} features {"->"} train</div>
                </button>
              </div>
              <div className="mt-2 text-xs text-mute">
                Updated: ingest {fmtTime(lastUpdated.ingestStatus)} | pipeline {fmtTime(lastUpdated.pipelineStatus)}
              </div>
              <div className="mt-6 grid gap-3 sm:grid-cols-2">
                <JsonBlock data={ingestStatus || {}} label="ingest status" />
                <JsonBlock data={pipelineStatus || {}} label="pipeline status" />
              </div>
            </div>

            <div className="rounded-2xl border border-line bg-panel/70 p-5">
              <h2 className="font-display text-xl">Operator Feed</h2>
              <div className="mt-4 space-y-2">
                {messages.length === 0 && <div className="text-sm text-mute">No events yet.</div>}
                {messages.map((m) => (
                  <div
                    key={m.id}
                    className={`rounded-lg border px-3 py-2 text-sm ${
                      m.tone === "err"
                        ? "border-bad/60 bg-bad/10 text-bad"
                        : m.tone === "ok"
                        ? "border-good/60 bg-good/10 text-good"
                        : "border-line bg-bg/50 text-mute"
                    }`}
                  >
                    {m.text}
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}

        {activeTab === "model" && (
          <section className="space-y-6">
            <div className="grid gap-3 md:grid-cols-4">
              <StatCard
                label="Accuracy vs Prev"
                value={latestModelRun?.accuracy != null ? `${(latestModelRun.accuracy * 100).toFixed(2)}%` : "n/a"}
                tone={(metricDelta(latestModelRun?.accuracy, previousModelRun?.accuracy) || 0) >= 0 ? "good" : "warn"}
              />
              <StatCard
                label="F1 vs Prev"
                value={latestModelRun?.f1 != null ? latestModelRun.f1.toFixed(3) : "n/a"}
                tone={(metricDelta(latestModelRun?.f1, previousModelRun?.f1) || 0) >= 0 ? "good" : "warn"}
              />
              <StatCard
                label="ROC AUC vs Prev"
                value={latestModelRun?.roc_auc != null ? latestModelRun.roc_auc.toFixed(3) : "n/a"}
                tone={(metricDelta(latestModelRun?.roc_auc, previousModelRun?.roc_auc) || 0) >= 0 ? "good" : "warn"}
              />
              <StatCard
                label="Recommended Hold"
                value={latestModelRun?.recommended_hold_threshold != null ? latestModelRun.recommended_hold_threshold.toFixed(2) : "n/a"}
                tone="accent"
              />
            </div>

            <div className="rounded-2xl border border-line bg-panel/70 p-5">
              <h2 className="font-display text-xl">Model Run History</h2>
              <div className="mt-2 text-xs text-mute">Updated: {fmtTime(lastUpdated.modelHistory)}</div>
              <div className="mt-4 overflow-auto rounded-lg border border-line">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-bg/90 text-mute">
                    <tr>
                      <th className="px-3 py-2 text-left">Run</th>
                      <th className="px-3 py-2 text-left">Feature Set</th>
                      <th className="px-3 py-2 text-right">Horizon</th>
                      <th className="px-3 py-2 text-right">Accuracy</th>
                      <th className="px-3 py-2 text-right">F1</th>
                      <th className="px-3 py-2 text-right">ROC AUC</th>
                      <th className="px-3 py-2 text-right">WF Acc</th>
                      <th className="px-3 py-2 text-right">Signals</th>
                    </tr>
                  </thead>
                  <tbody>
                    {modelHistory.map((row) => (
                      <tr key={row.version_id} className="border-t border-line/50">
                        <td className="px-3 py-2">{row.created_at ? fmtTime(row.created_at) : row.version_id}</td>
                        <td className="px-3 py-2">{row.training_feature_set}</td>
                        <td className="px-3 py-2 text-right">{row.horizon_days}d</td>
                        <td className="px-3 py-2 text-right">{row.accuracy != null ? row.accuracy.toFixed(3) : "-"}</td>
                        <td className="px-3 py-2 text-right">{row.f1 != null ? row.f1.toFixed(3) : "-"}</td>
                        <td className="px-3 py-2 text-right">{row.roc_auc != null ? row.roc_auc.toFixed(3) : "-"}</td>
                        <td className="px-3 py-2 text-right">{row.walk_forward_accuracy != null ? row.walk_forward_accuracy.toFixed(3) : "-"}</td>
                        <td className="px-3 py-2 text-right">{row.signals_count ?? "-"}</td>
                      </tr>
                    ))}
                    {modelHistory.length === 0 && (
                      <tr>
                        <td className="px-3 py-4 text-mute" colSpan={8}>No model runs logged yet.</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            <section className="grid gap-6 lg:grid-cols-[1fr_1fr]">
            <div className="rounded-2xl border border-line bg-panel/70 p-5">
              <h2 className="font-display text-xl">Predict</h2>
              <div className="mt-3 flex gap-2">
                <input
                  value={ticker}
                  onChange={(e) => setTicker(e.target.value.toUpperCase())}
                  className="w-40 rounded-lg border border-line bg-bg/70 px-3 py-2 text-sm outline-none focus:border-accent"
                />
                <button
                  className="rounded-lg border border-accent/60 bg-accent/20 px-4 py-2 text-sm hover:bg-accent/30"
                  onClick={() =>
                    runAction("predict", async () => {
                      const out = await api(`/predict?ticker=${encodeURIComponent(ticker)}`);
                      setPrediction(out);
                      await refreshAll();
                      return out;
                    })
                  }
                  title="GET /predict?ticker=..."
                >
                  Predict
                </button>
              </div>
              <div className="mt-2 text-xs text-mute">Updated: predict {fmtTime(lastUpdated.predictionLogs)} | model {fmtTime(lastUpdated.modelStatus)}</div>
              <div className="mt-4">
                <JsonBlock data={prediction || { hint: "Run prediction to view output" }} label="prediction output" />
              </div>
              <div className="mt-4">
                <JsonBlock data={modelStatus || {}} label="model status" />
              </div>
            </div>

            <div className="rounded-2xl border border-line bg-panel/70 p-5">
              <h2 className="font-display text-xl">Prediction Log</h2>
              <div className="mt-3 flex flex-wrap items-center gap-2 text-sm">
                <input
                  value={logTickerFilter}
                  onChange={(e) => setLogTickerFilter(e.target.value.toUpperCase())}
                  placeholder="Filter ticker"
                  className="w-36 rounded border border-line bg-bg/60 px-2 py-1 text-xs"
                />
                <select
                  value={logPredFilter}
                  onChange={(e) => setLogPredFilter(e.target.value)}
                  className="rounded border border-line bg-bg/60 px-2 py-1 text-xs"
                >
                  <option value="all">All</option>
                  <option value="up">Up</option>
                  <option value="down">Down</option>
                  <option value="hold">Hold</option>
                </select>
                <span className="text-xs text-mute">Rows: {filteredLogs.length}</span>
              </div>
              <div className="mt-3 max-h-[520px] overflow-auto rounded-lg border border-line">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-bg/90 text-mute">
                    <tr>
                      <th className="px-3 py-2 text-left">Time</th>
                      <th className="px-3 py-2 text-left">Ticker</th>
                      <th className="px-3 py-2 text-left">Pred</th>
                      <th className="px-3 py-2 text-right">Conf</th>
                      <th className="px-3 py-2 text-left">Model</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredLogs.map((row) => (
                      <tr key={row.id} className="border-t border-line/50">
                        <td className="px-3 py-2">{row.created_at || "-"}</td>
                        <td className="px-3 py-2">{row.ticker}</td>
                        <td className="px-3 py-2">{row.prediction}</td>
                        <td className="px-3 py-2 text-right">{Number(row.confidence || 0).toFixed(3)}</td>
                        <td className="px-3 py-2">{row.model_version || "-"}</td>
                      </tr>
                    ))}
                    {filteredLogs.length === 0 && (
                      <tr>
                        <td className="px-3 py-4 text-mute" colSpan={5}>
                          No prediction logs yet.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </section>
          </section>
        )}

        {activeTab === "quality" && (
          <section className="grid gap-6 lg:grid-cols-2">
            <div className="rounded-2xl border border-line bg-panel/70 p-5">
              <h2 className="font-display text-xl">Data Status</h2>
              <div className="mt-2 text-xs text-mute">Updated: {fmtTime(lastUpdated.dataStatus)}</div>
              <div className="mt-4">
                <JsonBlock data={dataStatus || {}} label="data status" />
              </div>
            </div>
            <div className="rounded-2xl border border-line bg-panel/70 p-5">
              <h2 className="font-display text-xl">Data Quality</h2>
              <div className="mt-2 text-xs text-mute">Updated: {fmtTime(lastUpdated.quality)}</div>
              <div className="mt-4">
                <JsonBlock data={quality || {}} label="data quality" />
              </div>
            </div>
          </section>
        )}

        {activeTab === "docs" && (
          <section className="grid gap-6 lg:grid-cols-2">
            <article className="rounded-2xl border border-line bg-panel/70 p-5">
              <h2 className="mb-3 font-display text-xl">README</h2>
              <div className="mb-3 text-xs text-mute">Updated: {fmtTime(lastUpdated.docs)}</div>
              <div className="prose prose-invert max-w-none text-sm prose-headings:font-display">
                <ReactMarkdown>{docs.readme_markdown || "_No README text loaded_"}</ReactMarkdown>
              </div>
            </article>
            <article className="rounded-2xl border border-line bg-panel/70 p-5">
              <h2 className="mb-3 font-display text-xl">Project Overview</h2>
              <pre className="max-h-[70vh] overflow-auto whitespace-pre-wrap rounded-lg border border-line bg-bg/60 p-3 text-xs text-mute">
                {docs.project_overview_text || "No overview loaded."}
              </pre>
            </article>
          </section>
        )}
      </div>
    </div>
  );
}
