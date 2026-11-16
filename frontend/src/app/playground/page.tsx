"use client";

import { useState } from "react";
import { api, CustomerFeatures, PredictionResponse } from "@/lib/api";
import {
  PlayCircle,
  Zap,
  CheckCircle,
  Sparkles,
  Database,
} from "lucide-react";

const PRESET_HIGH_RISK: CustomerFeatures = {
  customerID: "CUST-HIGH-RISK",
  gender: "Female",
  SeniorCitizen: 0,
  Partner: "No",
  Dependents: "No",
  tenure: 2,
  PhoneService: "Yes",
  MultipleLines: "No",
  InternetService: "Fiber optic",
  OnlineSecurity: "No",
  OnlineBackup: "No",
  DeviceProtection: "No",
  TechSupport: "No",
  StreamingTV: "Yes",
  StreamingMovies: "Yes",
  Contract: "Month-to-month",
  PaperlessBilling: "Yes",
  PaymentMethod: "Electronic check",
  MonthlyCharges: 98.5,
  TotalCharges: 197.0,
};

const PRESET_LOYAL: CustomerFeatures = {
  customerID: "CUST-LOYAL",
  gender: "Male",
  SeniorCitizen: 0,
  Partner: "Yes",
  Dependents: "Yes",
  tenure: 62,
  PhoneService: "Yes",
  MultipleLines: "Yes",
  InternetService: "DSL",
  OnlineSecurity: "Yes",
  OnlineBackup: "Yes",
  DeviceProtection: "Yes",
  TechSupport: "Yes",
  StreamingTV: "No",
  StreamingMovies: "No",
  Contract: "Two year",
  PaperlessBilling: "No",
  PaymentMethod: "Credit card (automatic)",
  MonthlyCharges: 54.2,
  TotalCharges: 3360.4,
};

const PRESET_DRIFTED: CustomerFeatures = {
  customerID: "CUST-DRIFT-FIBER",
  gender: "Female",
  SeniorCitizen: 1,
  Partner: "No",
  Dependents: "No",
  tenure: 5,
  PhoneService: "Yes",
  MultipleLines: "Yes",
  InternetService: "Fiber optic",
  OnlineSecurity: "No",
  OnlineBackup: "No",
  DeviceProtection: "No",
  TechSupport: "No",
  StreamingTV: "Yes",
  StreamingMovies: "Yes",
  Contract: "Month-to-month",
  PaperlessBilling: "Yes",
  PaymentMethod: "Electronic check",
  MonthlyCharges: 114.75,
  TotalCharges: 573.75,
};

export default function PredictionPlaygroundPage() {
  const [form, setForm] = useState<CustomerFeatures>(PRESET_HIGH_RISK);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchMsg, setBatchMsg] = useState<string | null>(null);
  const [labelSubmitted, setLabelSubmitted] = useState<boolean>(false);

  const handlePredict = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    try {
      setLoading(true);
      setLabelSubmitted(false);
      const res = await api.predict(form);
      setResult(res);
    } catch (err: any) {
      alert(`Prediction failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleSimulateBatch = async (scenario: "normal" | "drifted") => {
    try {
      setBatchLoading(true);
      setBatchMsg(null);
      const batchList: CustomerFeatures[] = [];
      for (let i = 0; i < 50; i++) {
        if (scenario === "drifted") {
          batchList.push({
            ...PRESET_DRIFTED,
            customerID: `PROD-DRIFT-${Math.floor(10000 + Math.random() * 90000)}`,
            tenure: Math.floor(1 + Math.random() * 8),
            MonthlyCharges: 95.0 + Math.random() * 30.0,
            TotalCharges: 300.0 + Math.random() * 500.0,
          });
        } else {
          batchList.push({
            ...PRESET_LOYAL,
            customerID: `PROD-NORM-${Math.floor(10000 + Math.random() * 90000)}`,
            tenure: Math.floor(10 + Math.random() * 50),
            MonthlyCharges: 40.0 + Math.random() * 40.0,
            TotalCharges: 1000.0 + Math.random() * 2000.0,
          });
        }
      }

      const res = await api.predictBatch(batchList);
      setBatchMsg(
        `Logged ${res.total_count} production inferences (Avg Probability: ${(res.average_churn_probability * 100).toFixed(1)}%).`
      );
    } catch (err: any) {
      alert(`Simulation failed: ${err.message}`);
    } finally {
      setBatchLoading(false);
    }
  };

  const handleDelayedLabel = async (actual: number) => {
    if (!result || !result.customer_id) return;
    try {
      await api.ingestLabels([
        {
          customer_id: result.customer_id,
          actual_churn: actual,
          prediction_log_id: result.prediction_id,
        },
      ]);
      setLabelSubmitted(true);
    } catch (err: any) {
      alert(`Failed to ingest label: ${err.message}`);
    }
  };

  return (
    <div className="space-y-8">
      {/* Editorial Header */}
      <div className="border-b border-border pb-6">
        <div className="text-xs uppercase tracking-widest text-muted font-mono font-medium mb-1">
          Interactive Inference
        </div>
        <h1 className="text-3xl font-serif tracking-tight text-primary flex items-center gap-3">
          Prediction Playground & Label Pairing
        </h1>
        <p className="text-muted mt-1.5 text-sm max-w-2xl leading-relaxed">
          Submit live customer account attributes to evaluate real-time churn probability, log inferences, and associate verified delayed outcomes.
        </p>
      </div>

      {/* Preset Profiles & Simulation Card */}
      <div className="bg-surface border border-border rounded-xl p-5 flex flex-wrap items-center justify-between gap-4 shadow-[0_1px_3px_rgba(0,0,0,0.03)]">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-terracotta" />
          <span className="text-xs text-muted font-medium">Archetypes:</span>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => setForm(PRESET_HIGH_RISK)}
              className="px-2.5 py-1 bg-background hover:bg-surface-hover border border-border text-xs rounded-lg text-primary font-medium transition-colors"
            >
              High Risk Churner
            </button>
            <button
              type="button"
              onClick={() => setForm(PRESET_LOYAL)}
              className="px-2.5 py-1 bg-background hover:bg-surface-hover border border-border text-xs rounded-lg text-primary font-medium transition-colors"
            >
              Loyal Customer
            </button>
            <button
              type="button"
              onClick={() => setForm(PRESET_DRIFTED)}
              className="px-2.5 py-1 bg-background hover:bg-surface-hover border border-border text-xs rounded-lg text-primary font-medium transition-colors"
            >
              Fiber Drift Profile
            </button>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Database className="h-4 w-4 text-muted" />
          <span className="text-xs text-muted">Traffic Simulator:</span>
          <button
            onClick={() => handleSimulateBatch("drifted")}
            disabled={batchLoading}
            className="px-3 py-1 bg-background hover:bg-surface-hover border border-border text-xs text-primary rounded-lg font-medium transition-colors disabled:opacity-50"
          >
            {batchLoading ? "Sending..." : "Simulate 50 Drifted"}
          </button>
          <button
            onClick={() => handleSimulateBatch("normal")}
            disabled={batchLoading}
            className="px-3 py-1 bg-background hover:bg-surface-hover border border-border text-xs text-primary rounded-lg font-medium transition-colors disabled:opacity-50"
          >
            {batchLoading ? "Sending..." : "Simulate 50 Normal"}
          </button>
        </div>
      </div>

      {batchMsg && (
        <div className="bg-terracotta/10 border border-terracotta/30 p-3 rounded-xl text-xs text-terracotta flex items-center justify-between">
          <span>{batchMsg}</span>
          <button onClick={() => setBatchMsg(null)} className="text-muted hover:text-primary">
            &times;
          </button>
        </div>
      )}

      {/* Main Grid: Form on Left, Output on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Input Form */}
        <form onSubmit={handlePredict} className="lg:col-span-2 bg-surface border border-border rounded-xl p-6 space-y-5 shadow-[0_1px_3px_rgba(0,0,0,0.03)]">
          <div className="flex items-center justify-between border-b border-border pb-3">
            <h2 className="text-xs uppercase tracking-wider font-mono text-muted font-medium">
              Customer Profile Attributes
            </h2>
            <span className="text-xs text-muted">19 Evaluated Inputs</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
            <div>
              <label className="block text-muted mb-1 font-mono text-[11px]">Customer ID</label>
              <input
                type="text"
                value={form.customerID || ""}
                onChange={(e) => setForm({ ...form, customerID: e.target.value })}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-primary font-mono focus:outline-none focus:border-terracotta transition-colors"
              />
            </div>

            <div>
              <label className="block text-muted mb-1 font-mono text-[11px]">Gender</label>
              <select
                value={form.gender}
                onChange={(e) => setForm({ ...form, gender: e.target.value })}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-primary focus:outline-none focus:border-terracotta transition-colors"
              >
                <option value="Male">Male</option>
                <option value="Female">Female</option>
              </select>
            </div>

            <div>
              <label className="block text-muted mb-1 font-mono text-[11px]">Senior Citizen</label>
              <select
                value={form.SeniorCitizen}
                onChange={(e) => setForm({ ...form, SeniorCitizen: parseInt(e.target.value) })}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-primary focus:outline-none focus:border-terracotta transition-colors"
              >
                <option value={0}>0 - No</option>
                <option value={1}>1 - Yes</option>
              </select>
            </div>

            <div>
              <label className="block text-muted mb-1 font-mono text-[11px]">Partner</label>
              <select
                value={form.Partner}
                onChange={(e) => setForm({ ...form, Partner: e.target.value })}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-primary focus:outline-none focus:border-terracotta transition-colors"
              >
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>

            <div>
              <label className="block text-muted mb-1 font-mono text-[11px]">Dependents</label>
              <select
                value={form.Dependents}
                onChange={(e) => setForm({ ...form, Dependents: e.target.value })}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-primary focus:outline-none focus:border-terracotta transition-colors"
              >
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>

            <div>
              <label className="block text-muted mb-1 font-mono text-[11px]">Tenure (Months: 0-72)</label>
              <input
                type="number"
                min={0}
                max={72}
                value={form.tenure}
                onChange={(e) => setForm({ ...form, tenure: parseInt(e.target.value) || 0 })}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-primary font-mono focus:outline-none focus:border-terracotta transition-colors"
              />
            </div>

            <div>
              <label className="block text-muted mb-1 font-mono text-[11px]">Contract Term</label>
              <select
                value={form.Contract}
                onChange={(e) => setForm({ ...form, Contract: e.target.value })}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-primary focus:outline-none focus:border-terracotta transition-colors"
              >
                <option value="Month-to-month">Month-to-month</option>
                <option value="One year">One year</option>
                <option value="Two year">Two year</option>
              </select>
            </div>

            <div>
              <label className="block text-muted mb-1 font-mono text-[11px]">Internet Service</label>
              <select
                value={form.InternetService}
                onChange={(e) => setForm({ ...form, InternetService: e.target.value })}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-primary focus:outline-none focus:border-terracotta transition-colors"
              >
                <option value="Fiber optic">Fiber optic</option>
                <option value="DSL">DSL</option>
                <option value="No">No Internet</option>
              </select>
            </div>

            <div>
              <label className="block text-muted mb-1 font-mono text-[11px]">Tech Support</label>
              <select
                value={form.TechSupport}
                onChange={(e) => setForm({ ...form, TechSupport: e.target.value })}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-primary focus:outline-none focus:border-terracotta transition-colors"
              >
                <option value="Yes">Yes</option>
                <option value="No">No</option>
                <option value="No internet service">No internet service</option>
              </select>
            </div>

            <div>
              <label className="block text-muted mb-1 font-mono text-[11px]">Online Security</label>
              <select
                value={form.OnlineSecurity}
                onChange={(e) => setForm({ ...form, OnlineSecurity: e.target.value })}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-primary focus:outline-none focus:border-terracotta transition-colors"
              >
                <option value="Yes">Yes</option>
                <option value="No">No</option>
                <option value="No internet service">No internet service</option>
              </select>
            </div>

            <div>
              <label className="block text-muted mb-1 font-mono text-[11px]">Payment Method</label>
              <select
                value={form.PaymentMethod}
                onChange={(e) => setForm({ ...form, PaymentMethod: e.target.value })}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-primary focus:outline-none focus:border-terracotta transition-colors"
              >
                <option value="Electronic check">Electronic check</option>
                <option value="Mailed check">Mailed check</option>
                <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
                <option value="Credit card (automatic)">Credit card (automatic)</option>
              </select>
            </div>

            <div>
              <label className="block text-muted mb-1 font-mono text-[11px]">Paperless Billing</label>
              <select
                value={form.PaperlessBilling}
                onChange={(e) => setForm({ ...form, PaperlessBilling: e.target.value })}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-primary focus:outline-none focus:border-terracotta transition-colors"
              >
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>

            <div>
              <label className="block text-muted mb-1 font-mono text-[11px]">Monthly Charges ($)</label>
              <input
                type="number"
                step="0.01"
                value={form.MonthlyCharges}
                onChange={(e) => setForm({ ...form, MonthlyCharges: parseFloat(e.target.value) || 0 })}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-primary font-mono focus:outline-none focus:border-terracotta transition-colors"
              />
            </div>

            <div>
              <label className="block text-muted mb-1 font-mono text-[11px]">Total Charges ($)</label>
              <input
                type="number"
                step="0.01"
                value={form.TotalCharges}
                onChange={(e) => setForm({ ...form, TotalCharges: parseFloat(e.target.value) || 0 })}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-primary font-mono focus:outline-none focus:border-terracotta transition-colors"
              />
            </div>

            <div>
              <label className="block text-muted mb-1 font-mono text-[11px]">Phone Service</label>
              <select
                value={form.PhoneService}
                onChange={(e) => setForm({ ...form, PhoneService: e.target.value })}
                className="w-full bg-background border border-border rounded-lg px-3 py-2 text-primary focus:outline-none focus:border-terracotta transition-colors"
              >
                <option value="Yes">Yes</option>
                <option value="No">No</option>
              </select>
            </div>
          </div>

          <div className="pt-4 border-t border-border flex justify-end">
            <button
              type="submit"
              disabled={loading}
              className="px-5 py-2.5 bg-terracotta hover:bg-terracotta-hover text-white font-medium text-xs rounded-xl shadow-sm transition-all flex items-center gap-2"
            >
              <Zap className="h-3.5 w-3.5" />
              <span>{loading ? "Computing Risk Score..." : "Run Inference"}</span>
            </button>
          </div>
        </form>

        {/* Live Result Display */}
        <div className="bg-surface border border-border rounded-xl p-6 flex flex-col justify-between shadow-[0_1px_3px_rgba(0,0,0,0.03)]">
          <div>
            <h2 className="text-xs uppercase tracking-wider font-mono text-muted font-medium border-b border-border pb-3 mb-4">
              Prediction Output & Probability
            </h2>

            {result ? (
              <div className="space-y-6">
                {/* Result Card */}
                <div
                  className={`p-5 rounded-xl border text-center ${
                    result.churn_prediction === 1
                      ? "bg-danger-muted border-danger/30 text-primary"
                      : "bg-success-muted border-success/30 text-primary"
                  }`}
                >
                  <div className="text-[11px] font-mono uppercase text-muted">
                    Model Assessment
                  </div>
                  <div className="text-xl font-serif mt-1">
                    {result.churn_prediction === 1 ? "High Churn Risk" : "Likely Retained"}
                  </div>
                  <div className="text-xs text-muted mt-1">
                    Classification: <span className="font-mono font-medium text-primary">{result.churn_label}</span>
                  </div>
                </div>

                {/* Probability Gauge Bar in Terracotta */}
                <div>
                  <div className="flex justify-between text-xs text-primary mb-1.5">
                    <span className="text-muted">Churn Probability</span>
                    <span className="font-mono font-medium text-terracotta">
                      {(result.churn_probability * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-background h-2 rounded-full overflow-hidden border border-border/50">
                    <div
                      className={`h-2 rounded-full transition-all duration-500 ${
                        result.churn_probability > 0.5 ? "bg-danger" : "bg-success"
                      }`}
                      style={{ width: `${result.churn_probability * 100}%` }}
                    />
                  </div>
                  <div className="flex justify-between text-[10px] text-muted mt-1 font-mono">
                    <span>0% (Safe)</span>
                    <span>Threshold: 50%</span>
                    <span>100% (Churn)</span>
                  </div>
                </div>

                {/* Metadata */}
                <div className="space-y-2 text-xs border-t border-border pt-4 text-muted">
                  <div className="flex justify-between">
                    <span>Model Version:</span>
                    <span className="font-mono text-primary">{result.model_version}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Inference Latency:</span>
                    <span className="font-mono text-primary">{result.latency_ms} ms</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Log ID:</span>
                    <span className="font-mono text-muted">{result.prediction_id || "N/A"}</span>
                  </div>
                </div>

                {/* Delayed Label Submission */}
                <div className="bg-background border border-border rounded-xl p-3.5 text-xs">
                  <div className="font-medium text-primary mb-1">Delayed Ground Truth Pairing:</div>
                  <p className="text-[11px] text-muted mb-3 leading-relaxed">
                    When this customer&apos;s actual outcome arrives, record it to update model performance metrics:
                  </p>

                  {labelSubmitted ? (
                    <div className="text-success font-medium flex items-center gap-1.5 text-xs">
                      <CheckCircle className="h-4 w-4" />
                      <span>Actual outcome recorded for performance monitoring!</span>
                    </div>
                  ) : (
                    <div className="flex gap-2">
                      <button
                        type="button"
                        onClick={() => handleDelayedLabel(1)}
                        className="flex-1 py-1.5 bg-danger-muted hover:bg-danger/20 border border-danger/30 text-danger rounded-lg text-xs font-medium transition-colors"
                      >
                        Confirm Churned (1)
                      </button>
                      <button
                        type="button"
                        onClick={() => handleDelayedLabel(0)}
                        className="flex-1 py-1.5 bg-success-muted hover:bg-success/20 border border-success/30 text-success rounded-lg text-xs font-medium transition-colors"
                      >
                        Confirm Retained (0)
                      </button>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="text-center py-16 text-muted text-xs">
                <PlayCircle className="h-10 w-10 text-muted/40 mx-auto mb-2" />
                Select a profile and click &quot;Run Inference&quot; to inspect real-time outputs.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
