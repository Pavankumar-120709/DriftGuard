"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api, MonitoringStatusResponse, ModelInfoResponse, PerformanceMetricsResponse } from "@/lib/api";
import {
  Activity,
  Cpu,
  GitBranch,
  ShieldAlert,
  ShieldCheck,
  TrendingUp,
  ArrowRight,
  Database,
  Clock,
  Layers,
} from "lucide-react";

export default function OverviewPage() {
  const [monitoring, setMonitoring] = useState<MonitoringStatusResponse | null>(null);
  const [champion, setChampion] = useState<ModelInfoResponse | null>(null);
  const [performance, setPerformance] = useState<PerformanceMetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        const [monData, champData, perfData] = await Promise.all([
          api.getMonitoringStatus().catch(() => null),
          api.getModelInfo().catch(() => null),
          api.getPerformanceMetrics().catch(() => null),
        ]);
        setMonitoring(monData);
        setChampion(champData);
        setPerformance(perfData);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  return (
    <div className="space-y-8">
      {/* Editorial Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 border-b border-border pb-6">
        <div>
          <div className="text-xs uppercase tracking-widest text-muted font-mono font-medium mb-1">
            Platform Observability
          </div>
          <h1 className="text-3xl font-serif tracking-tight text-primary">
            Continuous Machine Learning Lifecycle
          </h1>
          <p className="text-muted mt-1.5 text-sm max-w-2xl leading-relaxed">
            Automated churn risk evaluation, statistical data drift detection via Evidently AI, delayed-label concept monitoring, and controlled retraining gates.
          </p>
        </div>

        <div>
          <Link
            href="/playground"
            className="px-4 py-2.5 bg-terracotta hover:bg-terracotta-hover text-white font-medium text-xs rounded-xl shadow-sm transition-all flex items-center gap-2"
          >
            <span>Prediction Playground</span>
            <ArrowRight className="h-4 w-4" />
          </Link>
        </div>
      </div>

      {/* KPI Cards in Warm Surface */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
        {/* Card 1: Active Model */}
        <div className="bg-surface border border-border rounded-xl p-5 shadow-[0_1px_3px_rgba(0,0,0,0.03)] hover:border-terracotta/40 transition-colors">
          <div className="flex items-center justify-between text-muted mb-2">
            <span className="text-xs font-mono uppercase tracking-wider">Active Champion</span>
            <Cpu className="h-4 w-4 text-terracotta" />
          </div>
          <div className="text-xl font-medium text-primary truncate" title={champion?.version || "N/A"}>
            {champion ? champion.model_name.replace("_", " ") : "Loading..."}
          </div>
          <div className="text-xs font-mono text-terracotta mt-1 truncate">
            {champion?.version || "No champion registered"}
          </div>
          <div className="mt-4 pt-3 border-t border-border flex items-center justify-between text-xs text-muted">
            <span>Validation ROC-AUC</span>
            <span className="font-mono font-medium text-primary">
              {champion?.metrics?.val_roc_auc ? (champion.metrics.val_roc_auc * 100).toFixed(1) + "%" : "N/A"}
            </span>
          </div>
        </div>

        {/* Card 2: Data Drift Status */}
        <div className="bg-surface border border-border rounded-xl p-5 shadow-[0_1px_3px_rgba(0,0,0,0.03)] hover:border-terracotta/40 transition-colors">
          <div className="flex items-center justify-between text-muted mb-2">
            <span className="text-xs font-mono uppercase tracking-wider">Data Drift Status</span>
            <GitBranch className="h-4 w-4 text-terracotta" />
          </div>
          <div className="flex items-center gap-2 mt-1">
            {monitoring?.active_dataset_drift ? (
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
          <div className="text-xs text-muted mt-2">
            Drifted Features:{" "}
            <span className="font-mono text-primary font-medium">
              {monitoring?.latest_run
                ? `${monitoring.latest_run.drifted_features_count} / ${monitoring.latest_run.total_features_count}`
                : "0 / 0"}
            </span>{" "}
            ({((monitoring?.drift_share || 0) * 100).toFixed(1)}%)
          </div>
          <div className="mt-4 pt-3 border-t border-border flex items-center justify-between text-xs text-muted">
            <span>Threshold</span>
            <span className="font-mono text-primary">
              {((monitoring?.latest_run?.drift_share_threshold || 0.3) * 100).toFixed(0)}%
            </span>
          </div>
        </div>

        {/* Card 3: Production Inferences */}
        <div className="bg-surface border border-border rounded-xl p-5 shadow-[0_1px_3px_rgba(0,0,0,0.03)] hover:border-terracotta/40 transition-colors">
          <div className="flex items-center justify-between text-muted mb-2">
            <span className="text-xs font-mono uppercase tracking-wider">Production Volume</span>
            <Activity className="h-4 w-4 text-terracotta" />
          </div>
          <div className="text-2xl font-serif text-primary">
            {monitoring?.total_prediction_records?.toLocaleString() || 0}
          </div>
          <div className="text-xs text-muted mt-1">
            Logged inferences evaluated
          </div>
          <div className="mt-4 pt-3 border-t border-border flex items-center justify-between text-xs text-muted">
            <span>Diagnostics Completed</span>
            <span className="font-mono text-primary">
              {monitoring?.total_monitoring_runs || 0} runs
            </span>
          </div>
        </div>

        {/* Card 4: Production Ground Truth Performance */}
        <div className="bg-surface border border-border rounded-xl p-5 shadow-[0_1px_3px_rgba(0,0,0,0.03)] hover:border-terracotta/40 transition-colors">
          <div className="flex items-center justify-between text-muted mb-2">
            <span className="text-xs font-mono uppercase tracking-wider">Delayed-Label Accuracy</span>
            <TrendingUp className="h-4 w-4 text-success" />
          </div>
          <div className="text-2xl font-serif text-primary">
            {performance && performance.total_evaluated > 0
              ? (performance.accuracy * 100).toFixed(1) + "%"
              : "Awaiting"}
          </div>
          <div className="text-xs text-muted mt-1">
            Evaluated on{" "}
            <span className="font-mono text-primary font-medium">
              {performance?.total_evaluated || 0}
            </span>{" "}
            verified outcomes
          </div>
          <div className="mt-4 pt-3 border-t border-border flex items-center justify-between text-xs text-muted">
            <span>Protected Value</span>
            <span className="font-mono text-success font-medium">
              ${(performance?.estimated_net_benefit_usd || 0).toLocaleString()}
            </span>
          </div>
        </div>
      </div>

      {/* Lifecycle Flow with Minimal Editorial Aesthetic */}
      <div className="bg-surface border border-border rounded-xl p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-base font-serif font-medium text-primary flex items-center gap-2">
              <Layers className="h-4 w-4 text-terracotta" />
              <span>Closed-Loop MLOps Architecture</span>
            </h2>
            <p className="text-xs text-muted mt-0.5">
              Continuous validation, live inference logging, drift telemetry, and safe champion promotion
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-7 gap-3 text-center pt-2">
          <div className="bg-background border border-border rounded-xl p-3.5 hover:border-terracotta/40 transition-colors">
            <Database className="h-4 w-4 text-muted mx-auto mb-2" />
            <div className="text-xs font-medium text-primary">1. Ingestion</div>
            <div className="text-[11px] text-muted mt-1">Schema & Clean</div>
          </div>

          <div className="bg-background border border-border rounded-xl p-3.5 hover:border-terracotta/40 transition-colors">
            <Cpu className="h-4 w-4 text-muted mx-auto mb-2" />
            <div className="text-xs font-medium text-primary">2. Training</div>
            <div className="text-[11px] text-muted mt-1">LR, RF, XGBoost</div>
          </div>

          <div className="bg-background border border-border rounded-xl p-3.5 hover:border-terracotta/40 transition-colors">
            <Activity className="h-4 w-4 text-muted mx-auto mb-2" />
            <div className="text-xs font-medium text-primary">3. MLflow</div>
            <div className="text-[11px] text-muted mt-1">Artifact Tracking</div>
          </div>

          <div className="bg-background border border-border rounded-xl p-3.5 hover:border-terracotta/40 transition-colors">
            <TrendingUp className="h-4 w-4 text-muted mx-auto mb-2" />
            <div className="text-xs font-medium text-primary">4. Serving</div>
            <div className="text-[11px] text-muted mt-1">FastAPI Inference</div>
          </div>

          <div className="bg-background border border-border rounded-xl p-3.5 hover:border-terracotta/40 transition-colors">
            <GitBranch className="h-4 w-4 text-terracotta mx-auto mb-2" />
            <div className="text-xs font-medium text-primary">5. Evidently</div>
            <div className="text-[11px] text-muted mt-1">KS & Chi-sq Drift</div>
          </div>

          <div className="bg-background border border-border rounded-xl p-3.5 hover:border-terracotta/40 transition-colors">
            <Clock className="h-4 w-4 text-muted mx-auto mb-2" />
            <div className="text-xs font-medium text-primary">6. Delayed Labels</div>
            <div className="text-[11px] text-muted mt-1">True Outcomes</div>
          </div>

          <div className="bg-terracotta/10 border border-terracotta/30 rounded-xl p-3.5">
            <ShieldCheck className="h-4 w-4 text-terracotta mx-auto mb-2" />
            <div className="text-xs font-medium text-terracotta">7. Retraining</div>
            <div className="text-[11px] text-muted mt-1">Quality Gate</div>
          </div>
        </div>
      </div>

      {/* Two Column Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Model Key Indicators */}
        <div className="bg-surface border border-border rounded-xl p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-serif font-medium text-primary text-sm flex items-center gap-2">
              <Cpu className="h-4 w-4 text-terracotta" />
              <span>Champion Performance Baseline</span>
            </h3>
            <Link href="/models" className="text-xs text-terracotta hover:underline flex items-center gap-1 font-medium">
              View Registry <ArrowRight className="h-3 w-3" />
            </Link>
          </div>

          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="bg-background p-3 rounded-lg border border-border">
                <span className="text-muted block text-[11px]">Validation F1-Score</span>
                <span className="font-mono text-base font-medium text-primary">
                  {champion?.metrics?.val_f1 ? (champion.metrics.val_f1 * 100).toFixed(1) + "%" : "N/A"}
                </span>
              </div>
              <div className="bg-background p-3 rounded-lg border border-border">
                <span className="text-muted block text-[11px]">Test ROC-AUC</span>
                <span className="font-mono text-base font-medium text-primary">
                  {champion?.metrics?.test_roc_auc ? (champion.metrics.test_roc_auc * 100).toFixed(1) + "%" : "N/A"}
                </span>
              </div>
            </div>

            <div className="text-xs text-muted uppercase tracking-wider font-mono">
              Primary Predictive Drivers:
            </div>
            <div className="space-y-2.5">
              {champion?.feature_importance?.slice(0, 4).map((f) => (
                <div key={f.feature} className="text-xs">
                  <div className="flex justify-between text-primary mb-1">
                    <span className="font-mono text-muted">{f.feature}</span>
                    <span className="font-mono text-terracotta font-medium">{(f.importance * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-background rounded-full h-1.5 overflow-hidden border border-border/40">
                    <div
                      className="bg-terracotta h-1.5 rounded-full"
                      style={{ width: `${Math.min(100, f.importance * 220)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right: Monitoring & Safeguards */}
        <div className="bg-surface border border-border rounded-xl p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-serif font-medium text-primary text-sm flex items-center gap-2">
              <GitBranch className="h-4 w-4 text-terracotta" />
              <span>Monitoring & Retraining Gates</span>
            </h3>
            <Link href="/drift" className="text-xs text-terracotta hover:underline flex items-center gap-1 font-medium">
              Drift Telemetry <ArrowRight className="h-3 w-3" />
            </Link>
          </div>

          <div className="space-y-3.5 text-xs">
            <div className="flex items-center justify-between p-3.5 rounded-lg bg-background border border-border">
              <div>
                <div className="font-medium text-primary">Distribution Drift Gate</div>
                <div className="text-[11px] text-muted">Feature drift share exceeds 30%</div>
              </div>
              <div>
                {monitoring?.active_dataset_drift ? (
                  <span className="bg-danger-muted text-danger border border-danger/30 px-2.5 py-1 rounded-md font-medium text-[11px]">
                    Condition Met
                  </span>
                ) : (
                  <span className="bg-success-muted text-success border border-success/30 px-2.5 py-1 rounded-md font-medium text-[11px]">
                    Within Limits
                  </span>
                )}
              </div>
            </div>

            <div className="flex items-center justify-between p-3.5 rounded-lg bg-background border border-border">
              <div>
                <div className="font-medium text-primary">Performance Drop Gate</div>
                <div className="text-[11px] text-muted">Degradation &gt; 5% vs offline baseline</div>
              </div>
              <div>
                {performance?.performance_drop_detected ? (
                  <span className="bg-danger-muted text-danger border border-danger/30 px-2.5 py-1 rounded-md font-medium text-[11px]">
                    Drop Alert
                  </span>
                ) : (
                  <span className="bg-success-muted text-success border border-success/30 px-2.5 py-1 rounded-md font-medium text-[11px]">
                    Stable
                  </span>
                )}
              </div>
            </div>

            <div className="pt-2 flex gap-3">
              <Link
                href="/retraining"
                className="flex-1 text-center py-2.5 bg-background hover:bg-surface-hover text-primary border border-border rounded-xl font-medium transition-colors"
              >
                Inspect Retraining Pipeline
              </Link>
              <Link
                href="/drift"
                className="flex-1 text-center py-2.5 bg-terracotta/10 hover:bg-terracotta/20 text-terracotta border border-terracotta/30 rounded-xl font-medium transition-colors"
              >
                Run Drift Diagnostics
              </Link>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
