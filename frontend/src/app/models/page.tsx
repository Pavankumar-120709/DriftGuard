"use client";

import { useEffect, useState } from "react";
import {
  api,
  ModelInfoResponse,
  ChampionChallengerComparisonResponse,
} from "@/lib/api";
import {
  Cpu,
  GitCompare,
  History,
  ShieldCheck,
  Award,
  Layers,
} from "lucide-react";

export default function ModelsPage() {
  const [models, setModels] = useState<ModelInfoResponse[]>([]);
  const [comparison, setComparison] = useState<ChampionChallengerComparisonResponse | null>(null);
  const [promotions, setPromotions] = useState<any[]>([]);
  const [selectedChallenger, setSelectedChallenger] = useState<string>("");
  const [loading, setLoading] = useState(true);

  async function loadData(challengerVer?: string) {
    try {
      setLoading(true);
      const [modelsList, compData, promoData] = await Promise.all([
        api.listModels().catch(() => []),
        api.compareModels(challengerVer).catch(() => null),
        api.getPromotionsHistory().catch(() => []),
      ]);
      setModels(modelsList);
      setComparison(compData);
      setPromotions(promoData);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadData();
  }, []);

  const handleChallengerSelect = (ver: string) => {
    setSelectedChallenger(ver);
    loadData(ver);
  };

  return (
    <div className="space-y-8">
      {/* Editorial Header */}
      <div className="border-b border-border pb-6">
        <div className="text-xs uppercase tracking-widest text-muted font-mono font-medium mb-1">
          Governance & Versioning
        </div>
        <h1 className="text-3xl font-serif tracking-tight text-primary flex items-center gap-3">
          Model Registry & Quality Gates
        </h1>
        <p className="text-muted mt-1.5 text-sm max-w-2xl leading-relaxed">
          Centralized catalog of serialized pipeline artifacts, MLflow run linkages, and champion/challenger comparative evaluation.
        </p>
      </div>

      {/* Champion / Challenger Comparison Banner */}
      {comparison && comparison.champion && (
        <div className="bg-surface border border-border rounded-xl p-6 shadow-[0_1px_3px_rgba(0,0,0,0.03)]">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-base font-serif font-medium text-primary flex items-center gap-2">
              <GitCompare className="h-4 w-4 text-terracotta" />
              <span>Champion vs. Challenger Quality Gate</span>
            </h2>
            <div className="text-xs text-muted">
              Primary Gate: <span className="font-mono text-terracotta font-medium">{comparison.primary_metric}</span>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Champion Card */}
            <div className="bg-background border border-terracotta/30 rounded-xl p-5 relative">
              <div className="absolute top-4 right-4 flex items-center gap-1.5 bg-terracotta/15 text-terracotta border border-terracotta/30 text-xs px-2.5 py-0.5 rounded-full font-medium">
                <Award className="h-3.5 w-3.5" />
                <span>CHAMPION</span>
              </div>
              <div className="text-xs uppercase text-muted font-mono">Active Production Model</div>
              <div className="text-lg font-serif text-primary mt-1">
                {comparison.champion.model_name.replace("_", " ")}
              </div>
              <div className="text-xs font-mono text-muted mt-0.5">
                {comparison.champion.version}
              </div>

              <div className="grid grid-cols-3 gap-3 mt-4 pt-4 border-t border-border">
                <div>
                  <span className="text-[11px] text-muted block">Val ROC-AUC</span>
                  <span className="font-mono text-sm font-medium text-primary">
                    {comparison.champion.metrics?.val_roc_auc
                      ? (comparison.champion.metrics.val_roc_auc * 100).toFixed(2) + "%"
                      : "N/A"}
                  </span>
                </div>
                <div>
                  <span className="text-[11px] text-muted block">Val F1-Score</span>
                  <span className="font-mono text-sm font-medium text-primary">
                    {comparison.champion.metrics?.val_f1
                      ? (comparison.champion.metrics.val_f1 * 100).toFixed(2) + "%"
                      : "N/A"}
                  </span>
                </div>
                <div>
                  <span className="text-[11px] text-muted block">Test ROC-AUC</span>
                  <span className="font-mono text-sm font-medium text-primary">
                    {comparison.champion.metrics?.test_roc_auc
                      ? (comparison.champion.metrics.test_roc_auc * 100).toFixed(2) + "%"
                      : "N/A"}
                  </span>
                </div>
              </div>
            </div>

            {/* Challenger Card */}
            <div className="bg-background border border-border rounded-xl p-5 relative">
              <div className="absolute top-4 right-4 flex items-center gap-1.5 bg-surface text-muted border border-border text-xs px-2.5 py-0.5 rounded-full font-medium">
                <ShieldCheck className="h-3.5 w-3.5" />
                <span>CHALLENGER</span>
              </div>
              <div className="text-xs uppercase text-muted font-mono">Challenger Candidate</div>
              <div className="text-lg font-serif text-primary mt-1">
                {comparison.challenger
                  ? comparison.challenger.model_name.replace("_", " ")
                  : "No Challenger Selected"}
              </div>
              <div className="text-xs font-mono text-muted mt-0.5">
                {comparison.challenger?.version || "N/A"}
              </div>

              <div className="grid grid-cols-3 gap-3 mt-4 pt-4 border-t border-border">
                <div>
                  <span className="text-[11px] text-muted block">Val ROC-AUC</span>
                  <span className="font-mono text-sm font-medium text-primary">
                    {comparison.challenger?.metrics?.val_roc_auc
                      ? (comparison.challenger.metrics.val_roc_auc * 100).toFixed(2) + "%"
                      : "N/A"}
                  </span>
                </div>
                <div>
                  <span className="text-[11px] text-muted block">Val F1-Score</span>
                  <span className="font-mono text-sm font-medium text-primary">
                    {comparison.challenger?.metrics?.val_f1
                      ? (comparison.challenger.metrics.val_f1 * 100).toFixed(2) + "%"
                      : "N/A"}
                  </span>
                </div>
                <div>
                  <span className="text-[11px] text-muted block">Margin vs Champion</span>
                  <span
                    className={`font-mono text-sm font-medium ${
                      comparison.margin > 0 ? "text-success" : "text-muted"
                    }`}
                  >
                    {comparison.margin >= 0 ? `+${(comparison.margin * 100).toFixed(2)}%` : `${(comparison.margin * 100).toFixed(2)}%`}
                  </span>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-4 p-3 bg-background border border-border rounded-lg text-xs flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <span className="font-medium text-primary">Gate Evaluation:</span>
              <span className="text-muted">{comparison.recommendation}</span>
            </div>
            <div>
              {comparison.meets_threshold ? (
                <span className="bg-success-muted text-success border border-success/30 px-2 py-0.5 rounded font-medium text-[11px]">
                  Qualified for Promotion
                </span>
              ) : (
                <span className="bg-surface text-muted border border-border px-2 py-0.5 rounded font-medium text-[11px]">
                  Champion Retained
                </span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Model Registry Table */}
      <div className="bg-surface border border-border rounded-xl p-6 shadow-[0_1px_3px_rgba(0,0,0,0.03)]">
        <h2 className="text-base font-serif font-medium text-primary mb-4 flex items-center gap-2">
          <Layers className="h-4 w-4 text-terracotta" />
          <span>Registered Model Artifacts</span>
        </h2>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-border text-muted uppercase text-[11px] font-mono">
                <th className="py-3 px-4">Version</th>
                <th className="py-3 px-4">Architecture</th>
                <th className="py-3 px-4">Status</th>
                <th className="py-3 px-4">Val ROC-AUC</th>
                <th className="py-3 px-4">Val F1</th>
                <th className="py-3 px-4">Test ROC-AUC</th>
                <th className="py-3 px-4">MLflow Run ID</th>
                <th className="py-3 px-4">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border/60 font-mono">
              {models.map((m) => (
                <tr key={m.version} className="hover:bg-surface-hover/50 transition-colors">
                  <td className="py-3.5 px-4 font-medium text-primary">{m.version}</td>
                  <td className="py-3.5 px-4 text-primary font-sans">{m.model_name.replace("_", " ")}</td>
                  <td className="py-3.5 px-4 font-sans">
                    {m.is_champion ? (
                      <span className="bg-terracotta/15 text-terracotta border border-terracotta/30 px-2 py-0.5 rounded text-[11px] font-medium">
                        Active Champion
                      </span>
                    ) : (
                      <span className="bg-background text-muted border border-border px-2 py-0.5 rounded text-[11px]">
                        Registered
                      </span>
                    )}
                  </td>
                  <td className="py-3.5 px-4 text-primary">
                    {m.metrics?.val_roc_auc ? (m.metrics.val_roc_auc * 100).toFixed(2) + "%" : "N/A"}
                  </td>
                  <td className="py-3.5 px-4 text-primary">
                    {m.metrics?.val_f1 ? (m.metrics.val_f1 * 100).toFixed(2) + "%" : "N/A"}
                  </td>
                  <td className="py-3.5 px-4 text-primary">
                    {m.metrics?.test_roc_auc ? (m.metrics.test_roc_auc * 100).toFixed(2) + "%" : "N/A"}
                  </td>
                  <td className="py-3.5 px-4 text-muted truncate max-w-[140px]" title={m.mlflow_run_id || ""}>
                    {m.mlflow_run_id || "local"}
                  </td>
                  <td className="py-3.5 px-4 font-sans">
                    {!m.is_champion && (
                      <button
                        onClick={() => handleChallengerSelect(m.version)}
                        className="text-terracotta hover:underline text-xs font-medium"
                      >
                        Compare
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Promotion Audit Log */}
      <div className="bg-surface border border-border rounded-xl p-6 shadow-[0_1px_3px_rgba(0,0,0,0.03)]">
        <h2 className="text-base font-serif font-medium text-primary mb-4 flex items-center gap-2">
          <History className="h-4 w-4 text-terracotta" />
          <span>Champion Promotion & Rollback Audit History</span>
        </h2>

        {promotions.length === 0 ? (
          <div className="text-xs text-muted py-6 text-center">No promotions recorded yet.</div>
        ) : (
          <div className="space-y-3">
            {promotions.map((p) => (
              <div
                key={p.id}
                className="bg-background border border-border rounded-lg p-3.5 text-xs flex items-center justify-between"
              >
                <div>
                  <div className="flex items-center gap-2 font-mono text-primary font-medium">
                    <span className="text-terracotta">{p.promoted_version}</span>
                    <span className="text-muted font-sans text-[11px]">promoted over</span>
                    <span className="text-muted">{p.previous_version || "initial"}</span>
                  </div>
                  <div className="text-[11px] text-muted mt-1">{p.reason}</div>
                </div>
                <div className="text-right">
                  <span
                    className={`px-2 py-0.5 rounded text-[11px] font-medium ${
                      p.decision === "PROMOTED"
                        ? "bg-success-muted text-success border border-success/30"
                        : "bg-warning-muted text-warning border border-warning/30"
                    }`}
                  >
                    {p.decision}
                  </span>
                  <div className="text-[10px] text-muted mt-1 font-mono">
                    {new Date(p.timestamp).toLocaleString()}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
