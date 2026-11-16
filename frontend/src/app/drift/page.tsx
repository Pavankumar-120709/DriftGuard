"use client";

import { useEffect, useState } from "react";
import { api, MonitoringStatusResponse } from "@/lib/api";
import {
  GitBranch,
  ShieldAlert,
  ShieldCheck,
  RefreshCw,
  ExternalLink,
  Search,
  BarChart,
  HelpCircle,
} from "lucide-react";

export default function DriftMonitoringPage() {
  const [monitoring, setMonitoring] = useState<MonitoringStatusResponse | null>(null);
  const [runLoading, setRunLoading] = useState(false);
  const [filterText, setFilterText] = useState("");
  const [onlyDrifted, setOnlyDrifted] = useState(false);

  async function loadData() {
    try {
      const data = await api.getMonitoringStatus();
      setMonitoring(data);
    } catch (e) {
      console.error(e);
    }
  }

  useEffect(() => {
    loadData();
  }, []);

  const handleRunDrift = async () => {
    try {
      setRunLoading(true);
      await api.runDriftAnalysis(500, 0.05, 0.3);
      await loadData();
    } catch (err: any) {
      alert(`Drift run error: ${err.message}`);
    } finally {
      setRunLoading(false);
    }
  };

  const latest = monitoring?.latest_run;
  const features = latest?.features || [];

  const filteredFeatures = features.filter((f) => {
    const matchName = f.feature_name.toLowerCase().includes(filterText.toLowerCase());
    if (onlyDrifted) return matchName && f.is_drifted;
    return matchName;
  });

  return (
    <div className="space-y-8">
      {/* Editorial Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 border-b border-border pb-6">
        <div>
          <div className="text-xs uppercase tracking-widest text-muted font-mono font-medium mb-1">
            Statistical Telemetry
          </div>
          <h1 className="text-3xl font-serif tracking-tight text-primary flex items-center gap-3">
            Evidently AI Drift Monitoring
          </h1>
          <p className="text-muted mt-1.5 text-sm max-w-2xl leading-relaxed">
            Continuous population distribution monitoring comparing live production inferences against the baseline training dataset using two-sample KS and Chi-Square tests.
          </p>
        </div>

        <div className="flex items-center gap-3">
          {latest?.id && (
            <a
              href={`http://127.0.0.1:8000/monitoring/report/html/${latest.id}`}
              target="_blank"
              rel="noreferrer"
              className="px-3.5 py-2.5 bg-background hover:bg-surface-hover border border-border text-primary text-xs font-medium rounded-xl transition-colors flex items-center gap-2"
            >
              <span>View Evidently HTML Report</span>
              <ExternalLink className="h-3.5 w-3.5 text-muted" />
            </a>
          )}

          <button
            onClick={handleRunDrift}
            disabled={runLoading}
            className="px-4 py-2.5 bg-terracotta hover:bg-terracotta-hover disabled:opacity-50 text-white font-medium text-xs rounded-xl shadow-sm transition-all flex items-center gap-2"
          >
            <RefreshCw className={`h-4 w-4 ${runLoading ? "animate-spin" : ""}`} />
            <span>{runLoading ? "Analyzing Production Drift..." : "Run Drift Analysis"}</span>
          </button>
        </div>
      </div>

      {/* High-level Drift Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-surface border border-border rounded-xl p-5 shadow-[0_1px_3px_rgba(0,0,0,0.03)]">
          <span className="text-xs text-muted uppercase font-mono tracking-wider">Overall Status</span>
          <div className="mt-2 flex items-center gap-2">
            {latest?.dataset_drift_detected ? (
              <div className="flex items-center gap-1.5 text-danger font-medium text-sm bg-danger-muted px-2.5 py-1 rounded-lg border border-danger/20">
                <ShieldAlert className="h-4 w-4" />
                <span>Drift Flagged</span>
              </div>
            ) : (
              <div className="flex items-center gap-1.5 text-success font-medium text-sm bg-success-muted px-2.5 py-1 rounded-lg border border-success/20">
                <ShieldCheck className="h-4 w-4" />
                <span>Distribution Stable</span>
              </div>
            )}
          </div>
          <div className="text-[11px] text-muted mt-1.5">Evaluated against reference baseline</div>
        </div>

        <div className="bg-surface border border-border rounded-xl p-5 shadow-[0_1px_3px_rgba(0,0,0,0.03)]">
          <span className="text-xs text-muted uppercase font-mono tracking-wider">Drifted Feature Share</span>
          <div className="mt-2 text-2xl font-serif text-primary">
            {((latest?.drift_share || 0) * 100).toFixed(1)}%
          </div>
          <div className="text-[11px] text-muted mt-1">
            Threshold: {((latest?.drift_share_threshold || 0.3) * 100).toFixed(0)}%
          </div>
        </div>

        <div className="bg-surface border border-border rounded-xl p-5 shadow-[0_1px_3px_rgba(0,0,0,0.03)]">
          <span className="text-xs text-muted uppercase font-mono tracking-wider">Feature Drift Count</span>
          <div className="mt-2 text-2xl font-serif text-primary">
            {latest?.drifted_features_count || 0}{" "}
            <span className="text-xs text-muted font-sans font-normal">
              / {latest?.total_features_count || 0} features
            </span>
          </div>
          <div className="text-[11px] text-muted mt-1">Statistically significant changes</div>
        </div>

        <div className="bg-surface border border-border rounded-xl p-5 shadow-[0_1px_3px_rgba(0,0,0,0.03)]">
          <span className="text-xs text-muted uppercase font-mono tracking-wider">Production Volume</span>
          <div className="mt-2 text-2xl font-serif text-primary">
            {latest?.current_rows || 0}{" "}
            <span className="text-xs text-muted font-sans font-normal">
              vs {latest?.reference_rows || 0} ref
            </span>
          </div>
          <div className="text-[11px] text-muted mt-1">Inferences in monitoring window</div>
        </div>
      </div>

      {/* Conceptual Note in Claude Style */}
      <div className="bg-surface border border-border rounded-xl p-5 text-xs text-primary flex items-start gap-3.5 shadow-[0_1px_3px_rgba(0,0,0,0.03)]">
        <div className="w-8 h-8 rounded-lg bg-terracotta/15 flex items-center justify-center text-terracotta shrink-0 border border-terracotta/30">
          <HelpCircle className="h-4 w-4" />
        </div>
        <div className="space-y-1">
          <span className="font-serif font-medium text-sm text-primary">Core Distinction: Data Drift vs. Concept Drift</span>
          <p className="text-muted leading-relaxed text-xs">
            Data drift measures shifts in the input feature space \(P(X)\). While input distributions may change, a model can still maintain high predictive accuracy. Real model degradation (concept drift / \(P(Y|X)\) shift) can only be confirmed once actual delayed customer outcomes are ingested and verified against predictions.
          </p>
        </div>
      </div>

      {/* Feature Drift Detailed Table */}
      <div className="bg-surface border border-border rounded-xl p-6 shadow-[0_1px_3px_rgba(0,0,0,0.03)]">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-5">
          <h2 className="text-base font-serif font-medium text-primary flex items-center gap-2">
            <BarChart className="h-4 w-4 text-terracotta" />
            <span>Feature-Level Drift Diagnostics</span>
          </h2>

          <div className="flex items-center gap-3">
            <div className="relative">
              <Search className="h-4 w-4 text-muted absolute left-3 top-2.5" />
              <input
                type="text"
                placeholder="Search feature..."
                value={filterText}
                onChange={(e) => setFilterText(e.target.value)}
                className="bg-background border border-border rounded-lg pl-9 pr-3 py-1.5 text-xs text-primary focus:outline-none focus:border-terracotta transition-colors"
              />
            </div>

            <label className="flex items-center gap-1.5 text-xs text-muted cursor-pointer hover:text-primary transition-colors">
              <input
                type="checkbox"
                checked={onlyDrifted}
                onChange={(e) => setOnlyDrifted(e.target.checked)}
                className="rounded bg-background border-border text-terracotta focus:ring-terracotta/30"
              />
              <span>Only Drifted</span>
            </label>
          </div>
        </div>

        {filteredFeatures.length === 0 ? (
          <div className="text-center py-10 text-xs text-muted">
            No features match the filter or no drift run performed yet.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-border text-muted uppercase text-[11px] font-mono">
                  <th className="py-3 px-4">Feature Name</th>
                  <th className="py-3 px-4">Type</th>
                  <th className="py-3 px-4">Statistical Test</th>
                  <th className="py-3 px-4">Drift Score</th>
                  <th className="py-3 px-4">P-Value</th>
                  <th className="py-3 px-4">Status</th>
                  <th className="py-3 px-4">Distribution Telemetry</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/60 font-mono">
                {filteredFeatures.map((f) => (
                  <tr key={f.feature_name} className="hover:bg-surface-hover/50 transition-colors">
                    <td className="py-3.5 px-4 font-medium text-primary">{f.feature_name}</td>
                    <td className="py-3.5 px-4 text-muted font-sans">{f.feature_type}</td>
                    <td className="py-3.5 px-4 text-muted font-sans">{f.stat_test || "KS-Test"}</td>
                    <td className="py-3.5 px-4 text-terracotta font-medium">{f.drift_score.toFixed(4)}</td>
                    <td className="py-3.5 px-4 text-muted">
                      {f.p_value !== null && f.p_value !== undefined ? f.p_value.toFixed(4) : "N/A"}
                    </td>
                    <td className="py-3.5 px-4 font-sans">
                      {f.is_drifted ? (
                        <span className="bg-danger-muted text-danger border border-danger/30 px-2 py-0.5 rounded text-[11px] font-medium">
                          Drifted
                        </span>
                      ) : (
                        <span className="bg-success-muted text-success border border-success/30 px-2 py-0.5 rounded text-[11px] font-medium">
                          Stable
                        </span>
                      )}
                    </td>
                    <td className="py-3.5 px-4 text-[11px] text-muted">
                      {f.feature_type === "numerical" && f.reference_distribution && f.current_distribution ? (
                        <span>
                          Ref \(\mu\): {f.reference_distribution.mean} &rarr; Prod \(\mu\):{" "}
                          <span className={f.is_drifted ? "text-danger font-medium" : "text-primary"}>
                            {f.current_distribution.mean}
                          </span>
                        </span>
                      ) : (
                        <span className="truncate max-w-[200px] inline-block font-sans">
                          Categorical distribution tracked
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
