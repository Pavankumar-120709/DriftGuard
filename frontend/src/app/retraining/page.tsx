"use client";

import { useEffect, useState } from "react";
import { api, RetrainingJobResponse } from "@/lib/api";
import {
  RefreshCw,
  RotateCcw,
  ShieldCheck,
  Clock,
  CheckCircle,
  AlertTriangle,
} from "lucide-react";

export default function RetrainingPage() {
  const [history, setHistory] = useState<RetrainingJobResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const [rollbackLoading, setRollbackLoading] = useState(false);
  const [triggerReason, setTriggerReason] = useState("Manual retraining triggered from dashboard");
  const [minImprovement, setMinImprovement] = useState(0.005);
  const [statusMsg, setStatusMsg] = useState<{ type: "success" | "error"; text: string } | null>(null);

  async function loadHistory() {
    try {
      const data = await api.getRetrainingHistory();
      setHistory(data);
    } catch (e) {
      console.error(e);
    }
  }

  useEffect(() => {
    loadHistory();
  }, []);

  const handleTriggerRetrain = async () => {
    try {
      setLoading(true);
      setStatusMsg(null);
      const res = await api.triggerRetraining(triggerReason, minImprovement);
      setStatusMsg({
        type: "success",
        text: `Retraining finished with decision: ${res.decision}. Reason: ${res.decision_reason}`,
      });
      await loadHistory();
    } catch (err: any) {
      setStatusMsg({
        type: "error",
        text: `Retraining failed: ${err.message}`,
      });
    } finally {
      setLoading(false);
    }
  };

  const handleRollback = async () => {
    if (!confirm("Are you sure you want to rollback the active champion to the previous model version?")) {
      return;
    }
    try {
      setRollbackLoading(true);
      setStatusMsg(null);
      const res = await api.rollbackModel();
      if (res.success) {
        setStatusMsg({
          type: "success",
          text: `${res.message} Current champion: ${res.current_champion}`,
        });
      } else {
        setStatusMsg({
          type: "error",
          text: `Rollback failed: ${res.message}`,
        });
      }
      await loadHistory();
    } catch (err: any) {
      setStatusMsg({
        type: "error",
        text: `Rollback error: ${err.message}`,
      });
    } finally {
      setRollbackLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Editorial Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 border-b border-border pb-6">
        <div>
          <div className="text-xs uppercase tracking-widest text-muted font-mono font-medium mb-1">
            Automated Governance
          </div>
          <h1 className="text-3xl font-serif tracking-tight text-primary flex items-center gap-3">
            Controlled Model Retraining & Quality Gates
          </h1>
          <p className="text-muted mt-1.5 text-sm max-w-2xl leading-relaxed">
            Multi-model retraining workflow protected by atomic execution locks, baseline safety floors, and challenger promotion evaluation.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleRollback}
            disabled={rollbackLoading || loading}
            className="px-3.5 py-2.5 bg-background hover:bg-surface-hover border border-border text-primary font-medium text-xs rounded-xl transition-colors flex items-center gap-2 disabled:opacity-50"
          >
            <RotateCcw className={`h-3.5 w-3.5 text-terracotta ${rollbackLoading ? "animate-spin" : ""}`} />
            <span>Rollback Champion</span>
          </button>
        </div>
      </div>

      {statusMsg && (
        <div
          className={`p-4 rounded-xl border text-xs flex items-center gap-2.5 ${
            statusMsg.type === "success"
              ? "bg-success-muted border-success/30 text-success"
              : "bg-danger-muted border-danger/30 text-danger"
          }`}
        >
          {statusMsg.type === "success" ? (
            <CheckCircle className="h-4 w-4 shrink-0" />
          ) : (
            <AlertTriangle className="h-4 w-4 shrink-0" />
          )}
          <span className="font-medium">{statusMsg.text}</span>
        </div>
      )}

      {/* Retraining Controls & Safeguards Card */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1 bg-surface border border-border rounded-xl p-6 space-y-4 shadow-[0_1px_3px_rgba(0,0,0,0.03)]">
          <h2 className="text-xs uppercase tracking-wider font-mono text-muted font-medium border-b border-border pb-3">
            Trigger Retraining Pipeline
          </h2>

          <div className="text-xs space-y-3.5">
            <div>
              <label className="block text-muted mb-1 font-mono text-[11px]">Audit Justification</label>
              <textarea
                value={triggerReason}
                onChange={(e) => setTriggerReason(e.target.value)}
                rows={2}
                className="w-full bg-background border border-border rounded-lg p-2.5 text-primary focus:outline-none focus:border-terracotta transition-colors"
              />
            </div>

            <div>
              <label className="block text-muted mb-1 font-mono text-[11px]">
                Min Improvement Margin (\(\Delta\) ROC-AUC)
              </label>
              <input
                type="number"
                step="0.001"
                value={minImprovement}
                onChange={(e) => setMinImprovement(parseFloat(e.target.value) || 0.001)}
                className="w-full bg-background border border-border rounded-lg px-3 py-1.5 text-primary font-mono focus:outline-none focus:border-terracotta transition-colors"
              />
              <span className="text-[10px] text-muted block mt-1">
                Challenger must beat champion by at least this margin to be promoted.
              </span>
            </div>

            <button
              onClick={handleTriggerRetrain}
              disabled={loading}
              className="w-full py-2.5 bg-terracotta hover:bg-terracotta-hover disabled:opacity-50 text-white font-medium text-xs rounded-xl shadow-sm transition-all flex items-center justify-center gap-2"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
              <span>{loading ? "Retraining Models..." : "Execute Retraining"}</span>
            </button>
          </div>
        </div>

        {/* Safeguards Card */}
        <div className="lg:col-span-2 bg-surface border border-border rounded-xl p-6 shadow-[0_1px_3px_rgba(0,0,0,0.03)]">
          <h2 className="text-xs uppercase tracking-wider font-mono text-muted font-medium border-b border-border pb-3 mb-4">
            Production Safety Invariants
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            <div className="bg-background p-4 rounded-xl border border-border">
              <div className="flex items-center gap-2 font-medium text-primary mb-1">
                <ShieldCheck className="h-4 w-4 text-terracotta" />
                <span>Quality Gate Evaluation</span>
              </div>
              <p className="text-muted text-[11px] leading-relaxed">
                Retraining is never promoted automatically. A challenger model is trained on validated data, logged to MLflow, and evaluated against the current champion.
              </p>
            </div>

            <div className="bg-background p-4 rounded-xl border border-border">
              <div className="flex items-center gap-2 font-medium text-primary mb-1">
                <ShieldCheck className="h-4 w-4 text-terracotta" />
                <span>Baseline Quality Floor</span>
              </div>
              <p className="text-muted text-[11px] leading-relaxed">
                Challengers must meet minimum safety thresholds (ROC-AUC &ge; 0.70, F1 &ge; 0.45) to ensure safety even if the active champion has degraded.
              </p>
            </div>

            <div className="bg-background p-4 rounded-xl border border-border">
              <div className="flex items-center gap-2 font-medium text-primary mb-1">
                <ShieldCheck className="h-4 w-4 text-terracotta" />
                <span>Atomic Concurrency Lock</span>
              </div>
              <p className="text-muted text-[11px] leading-relaxed">
                An atomic execution lock prevents race conditions and concurrent retraining runs from conflicting over the model registry.
              </p>
            </div>

            <div className="bg-background p-4 rounded-xl border border-border">
              <div className="flex items-center gap-2 font-medium text-primary mb-1">
                <RotateCcw className="h-4 w-4 text-terracotta" />
                <span>One-Click Champion Rollback</span>
              </div>
              <p className="text-muted text-[11px] leading-relaxed">
                If an unexpected production edge-case arises, the previous champion model artifact is preserved and can be instantly reinstated.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Retraining History Table */}
      <div className="bg-surface border border-border rounded-xl p-6 shadow-[0_1px_3px_rgba(0,0,0,0.03)]">
        <h2 className="text-base font-serif font-medium text-primary mb-4 flex items-center gap-2">
          <Clock className="h-4 w-4 text-terracotta" />
          <span>Retraining Execution & Decision Audit Trail</span>
        </h2>

        {history.length === 0 ? (
          <div className="text-center py-8 text-xs text-muted">
            No retraining runs recorded yet.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-border text-muted uppercase text-[11px] font-mono">
                  <th className="py-3 px-4">Triggered</th>
                  <th className="py-3 px-4">Status</th>
                  <th className="py-3 px-4">Decision</th>
                  <th className="py-3 px-4">Challenger Version</th>
                  <th className="py-3 px-4">Champion Version</th>
                  <th className="py-3 px-4">Decision Reason</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/60 font-mono">
                {history.map((j) => (
                  <tr key={j.id} className="hover:bg-surface-hover/50 transition-colors">
                    <td className="py-3.5 px-4 text-muted">
                      {new Date(j.triggered_at).toLocaleString()}
                    </td>
                    <td className="py-3.5 px-4 font-sans">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-medium ${
                          j.status === "COMPLETED"
                            ? "bg-success-muted text-success border border-success/30"
                            : j.status === "RUNNING"
                            ? "bg-terracotta/15 text-terracotta border border-terracotta/30 animate-pulse"
                            : "bg-danger-muted text-danger border border-danger/30"
                        }`}
                      >
                        {j.status}
                      </span>
                    </td>
                    <td className="py-3.5 px-4 font-sans">
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-medium ${
                          j.decision === "PROMOTED"
                            ? "bg-success-muted text-success border border-success/30"
                            : j.decision === "REJECTED"
                            ? "bg-danger-muted text-danger border border-danger/30"
                            : "bg-background text-muted border border-border"
                        }`}
                      >
                        {j.decision}
                      </span>
                    </td>
                    <td className="py-3.5 px-4 text-primary">
                      {j.challenger_version || "N/A"}
                    </td>
                    <td className="py-3.5 px-4 text-muted">
                      {j.champion_version || "N/A"}
                    </td>
                    <td className="py-3.5 px-4 text-muted max-w-[280px] truncate font-sans" title={j.decision_reason || ""}>
                      {j.decision_reason || j.trigger_reason}
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
