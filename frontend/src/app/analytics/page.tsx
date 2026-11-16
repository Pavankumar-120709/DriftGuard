"use client";

import { useEffect, useState } from "react";
import {
  api,
  ModelInfoResponse,
  PerformanceMetricsResponse,
  MonitoringStatusResponse,
} from "@/lib/api";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
} from "recharts";
import { BarChart2, DollarSign, Activity, TrendingUp } from "lucide-react";

export default function AnalyticsPage() {
  const [champion, setChampion] = useState<ModelInfoResponse | null>(null);
  const [performance, setPerformance] = useState<PerformanceMetricsResponse | null>(null);
  const [monitoring, setMonitoring] = useState<MonitoringStatusResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        const [c, p, m] = await Promise.all([
          api.getModelInfo().catch(() => null),
          api.getPerformanceMetrics().catch(() => null),
          api.getMonitoringStatus().catch(() => null),
        ]);
        setChampion(c);
        setPerformance(p);
        setMonitoring(m);
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  // Format feature importance for Recharts
  const featData =
    champion?.feature_importance?.slice(0, 8).map((f) => ({
      feature: f.feature.length > 15 ? f.feature.substring(0, 15) + "..." : f.feature,
      importance: Number((f.importance * 100).toFixed(1)),
    })) || [];

  // Metrics comparison data
  const metricsComparisonData = [
    {
      metric: "Accuracy",
      Validation: Number(((champion?.metrics?.val_accuracy || 0) * 100).toFixed(1)),
      Production: Number(((performance?.accuracy || 0) * 100).toFixed(1)),
    },
    {
      metric: "Precision",
      Validation: Number(((champion?.metrics?.val_precision || 0) * 100).toFixed(1)),
      Production: Number(((performance?.precision || 0) * 100).toFixed(1)),
    },
    {
      metric: "Recall",
      Validation: Number(((champion?.metrics?.val_recall || 0) * 100).toFixed(1)),
      Production: Number(((performance?.recall || 0) * 100).toFixed(1)),
    },
    {
      metric: "F1 Score",
      Validation: Number(((champion?.metrics?.val_f1 || 0) * 100).toFixed(1)),
      Production: Number(((performance?.f1 || 0) * 100).toFixed(1)),
    },
    {
      metric: "ROC-AUC",
      Validation: Number(((champion?.metrics?.val_roc_auc || 0) * 100).toFixed(1)),
      Production: Number(((performance?.roc_auc || 0) * 100).toFixed(1)),
    },
  ];

  return (
    <div className="space-y-8">
      {/* Editorial Header */}
      <div className="border-b border-border pb-6">
        <div className="text-xs uppercase tracking-widest text-muted font-mono font-medium mb-1">
          Quantitative Evaluation
        </div>
        <h1 className="text-3xl font-serif tracking-tight text-primary flex items-center gap-3">
          Performance Telemetry & Driver Analytics
        </h1>
        <p className="text-muted mt-1.5 text-sm max-w-2xl leading-relaxed">
          Comprehensive comparison of offline validation baselines against live delayed-label outcomes, global predictive drivers, and estimated business benefit.
        </p>
      </div>

      {/* Top Cards: Financial Impact & Production Verification */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <div className="bg-surface border border-border rounded-xl p-5 shadow-[0_1px_3px_rgba(0,0,0,0.03)]">
          <div className="flex items-center justify-between text-muted mb-2">
            <span className="text-xs uppercase tracking-wider font-mono">Net Protected Value</span>
            <DollarSign className="h-4 w-4 text-success" />
          </div>
          <div className="text-2xl font-serif text-success">
            ${(performance?.estimated_net_benefit_usd || 0).toLocaleString()}
          </div>
          <div className="text-xs text-muted mt-1">
            Customer lifetime value saved minus intervention costs
          </div>
        </div>

        <div className="bg-surface border border-border rounded-xl p-5 shadow-[0_1px_3px_rgba(0,0,0,0.03)]">
          <div className="flex items-center justify-between text-muted mb-2">
            <span className="text-xs uppercase tracking-wider font-mono">Verified Samples</span>
            <Activity className="h-4 w-4 text-terracotta" />
          </div>
          <div className="text-2xl font-serif text-primary">
            {performance?.total_evaluated || 0}
          </div>
          <div className="text-xs text-muted mt-1">
            Joined inferences with delayed actual outcomes
          </div>
        </div>

        <div className="bg-surface border border-border rounded-xl p-5 shadow-[0_1px_3px_rgba(0,0,0,0.03)]">
          <div className="flex items-center justify-between text-muted mb-2">
            <span className="text-xs uppercase tracking-wider font-mono">Performance Gate</span>
            <TrendingUp className="h-4 w-4 text-terracotta" />
          </div>
          <div className="text-sm font-medium mt-1">
            {performance?.performance_drop_detected ? (
              <span className="text-danger bg-danger-muted px-2.5 py-1 rounded-md border border-danger/20 font-medium">
                Degradation Detected (&gt;5%)
              </span>
            ) : (
              <span className="text-success bg-success-muted px-2.5 py-1 rounded-md border border-success/20 font-medium">
                Within Tolerance (&le;5%)
              </span>
            )}
          </div>
          <div className="text-xs text-muted mt-2">
            Monitored against champion baseline
          </div>
        </div>
      </div>

      {/* Two Column Charts: Metrics Comparison & Feature Importance */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Validation vs Production Comparison */}
        <div className="bg-surface border border-border rounded-xl p-6 shadow-[0_1px_3px_rgba(0,0,0,0.03)]">
          <h2 className="text-base font-serif font-medium text-primary mb-1">
            Validation Baseline vs. Live Production
          </h2>
          <p className="text-xs text-muted mb-6">
            Compare offline validation scores against verified live production outcomes (%)
          </p>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={metricsComparisonData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(139, 134, 125, 0.15)" />
                <XAxis dataKey="metric" stroke="var(--muted)" fontSize={11} tickLine={false} />
                <YAxis stroke="var(--muted)" fontSize={11} domain={[0, 100]} tickLine={false} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "var(--surface)",
                    borderColor: "var(--border)",
                    color: "var(--foreground)",
                    fontSize: "12px",
                    borderRadius: "8px",
                  }}
                />
                <Legend wrapperStyle={{ fontSize: "12px", paddingTop: "12px" }} />
                <Bar dataKey="Validation" fill="#D97757" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Production" fill="#54836A" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Feature Importance Drivers */}
        <div className="bg-surface border border-border rounded-xl p-6 shadow-[0_1px_3px_rgba(0,0,0,0.03)]">
          <h2 className="text-base font-serif font-medium text-primary mb-1">
            Primary Predictive Drivers (SHAP / Global Weights)
          </h2>
          <p className="text-xs text-muted mb-6">
            Relative importance contributing to customer churn classification (%)
          </p>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={featData}
                layout="vertical"
                margin={{ top: 10, right: 20, left: 20, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(139, 134, 125, 0.15)" />
                <XAxis type="number" stroke="var(--muted)" fontSize={11} domain={[0, "auto"]} tickLine={false} />
                <YAxis dataKey="feature" type="category" stroke="var(--muted)" fontSize={10} width={90} tickLine={false} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "var(--surface)",
                    borderColor: "var(--border)",
                    color: "var(--foreground)",
                    fontSize: "12px",
                    borderRadius: "8px",
                  }}
                />
                <Bar dataKey="importance" fill="#C18A48" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Production Confusion Matrix Visualizer in Warm Tones */}
      <div className="bg-surface border border-border rounded-xl p-6 shadow-[0_1px_3px_rgba(0,0,0,0.03)]">
        <h2 className="text-base font-serif font-medium text-primary mb-1">
          Production Outcome Matrix
        </h2>
        <p className="text-xs text-muted mb-6">
          Calculated strictly on verified delayed customer churn outcomes
        </p>

        <div className="max-w-md mx-auto grid grid-cols-2 gap-4 text-center">
          <div className="bg-background border border-border rounded-xl p-4">
            <div className="text-[11px] font-mono text-muted uppercase">True Negatives (TN)</div>
            <div className="text-2xl font-serif text-success mt-1">
              {performance?.true_negatives || 0}
            </div>
            <div className="text-[11px] text-muted mt-1">Correctly identified retained customers</div>
          </div>

          <div className="bg-background border border-border rounded-xl p-4">
            <div className="text-[11px] font-mono text-muted uppercase">False Positives (FP)</div>
            <div className="text-2xl font-serif text-warning mt-1">
              {performance?.false_positives || 0}
            </div>
            <div className="text-[11px] text-muted mt-1">Unnecessary retention incentives ($120/ea)</div>
          </div>

          <div className="bg-background border border-border rounded-xl p-4">
            <div className="text-[11px] font-mono text-muted uppercase">False Negatives (FN)</div>
            <div className="text-2xl font-serif text-danger mt-1">
              {performance?.false_negatives || 0}
            </div>
            <div className="text-[11px] text-muted mt-1">Uncaught churners ($1,500 lost CLV)</div>
          </div>

          <div className="bg-background border border-border rounded-xl p-4">
            <div className="text-[11px] font-mono text-muted uppercase">True Positives (TP)</div>
            <div className="text-2xl font-serif text-terracotta mt-1">
              {performance?.true_positives || 0}
            </div>
            <div className="text-[11px] text-muted mt-1">Successfully intercepted churners</div>
          </div>
        </div>
      </div>
    </div>
  );
}
