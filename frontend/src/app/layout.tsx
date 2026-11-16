import type { Metadata } from "next";
import "./globals.css";
import { Navbar } from "@/components/Navbar";

export const metadata: Metadata = {
  title: "DriftGuard — End-to-End MLOps Platform",
  description: "Customer Churn Prediction with Evidently AI Drift Detection and Controlled Retraining",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-background text-primary flex flex-col antialiased selection:bg-terracotta/20 selection:text-terracotta">
        <Navbar />
        <main className="flex-1 max-w-7xl w-full mx-auto p-6 md:p-8 space-y-8">{children}</main>
        <footer className="border-t border-border py-6 px-6 text-center text-xs text-muted">
          DriftGuard MLOps Platform &bull; Warm Minimal Design &bull; MLflow Tracking &bull; Evidently AI &bull; FastAPI
        </footer>
      </body>
    </html>
  );
}
