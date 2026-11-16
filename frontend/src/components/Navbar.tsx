"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, useEffect } from "react";
import { api } from "@/lib/api";
import { ThemeToggle } from "./ThemeToggle";
import {
  Activity,
  Cpu,
  GitBranch,
  PlayCircle,
  RefreshCw,
  BarChart2,
  ShieldCheck,
  AlertTriangle,
} from "lucide-react";

export function Navbar() {
  const pathname = usePathname();
  const [championVer, setChampionVer] = useState<string>("Loading...");
  const [driftDetected, setDriftDetected] = useState<boolean>(false);
  const [isHealthy, setIsHealthy] = useState<boolean>(true);

  useEffect(() => {
    async function loadStatus() {
      try {
        const health = await api.getHealth();
        setIsHealthy(health.status === "healthy");
        if (health.champion_version) {
          setChampionVer(health.champion_version);
        }
        const mon = await api.getMonitoringStatus();
        setDriftDetected(mon.active_dataset_drift);
      } catch (e) {
        setIsHealthy(false);
        setChampionVer("Offline");
      }
    }
    loadStatus();
    const interval = setInterval(loadStatus, 15000);
    return () => clearInterval(interval);
  }, []);

  const navLinks = [
    { href: "/", label: "Overview", icon: Activity },
    { href: "/models", label: "Model Registry", icon: Cpu },
    { href: "/drift", label: "Drift Monitoring", icon: GitBranch },
    { href: "/playground", label: "Playground", icon: PlayCircle },
    { href: "/retraining", label: "Retraining", icon: RefreshCw },
    { href: "/analytics", label: "Analytics", icon: BarChart2 },
  ];

  return (
    <nav className="border-b border-border bg-sidebar/80 backdrop-blur-sm sticky top-0 z-50 px-6 py-3.5 transition-colors">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <div className="flex items-center space-x-8">
          <Link href="/" className="flex items-center space-x-2.5 group">
            <div className="w-8 h-8 rounded-lg bg-terracotta/15 flex items-center justify-center text-terracotta border border-terracotta/30 group-hover:border-terracotta transition-colors">
              <ShieldCheck className="h-5 w-5" />
            </div>
            <div className="flex items-baseline space-x-1.5">
              <span className="font-semibold text-lg tracking-tight text-primary">
                DriftGuard
              </span>
              <span className="text-[10px] font-mono uppercase tracking-widest text-muted">
                MLOps
              </span>
            </div>
          </Link>

          <div className="hidden md:flex items-center space-x-1">
            {navLinks.map((link) => {
              const Icon = link.icon;
              const isActive = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-sm transition-all ${
                    isActive
                      ? "bg-surface text-terracotta font-medium shadow-sm border border-border"
                      : "text-muted hover:text-primary hover:bg-surface/50"
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  <span>{link.label}</span>
                </Link>
              );
            })}
          </div>
        </div>

        <div className="flex items-center space-x-3">
          {driftDetected && (
            <div className="flex items-center space-x-1.5 text-xs bg-danger-muted text-danger border border-danger/30 px-2.5 py-1 rounded-full font-medium">
              <AlertTriangle className="h-3.5 w-3.5" />
              <span>Drift Alert</span>
            </div>
          )}

          <div className="hidden sm:flex items-center space-x-2 text-xs bg-surface border border-border px-3 py-1.5 rounded-lg">
            <span className="text-muted">Champion:</span>
            <span className="font-mono text-primary font-medium">{championVer}</span>
            <span
              className={`h-2 w-2 rounded-full ${isHealthy ? "bg-success" : "bg-danger"}`}
              title={isHealthy ? "API Connected" : "Connection Error"}
            />
          </div>

          <ThemeToggle />
        </div>
      </div>
    </nav>
  );
}
