"use client";

import { useEffect, useState } from "react";
import { Sun, Moon } from "lucide-react";

export function ThemeToggle() {
  const [isDark, setIsDark] = useState<boolean>(false);
  const [mounted, setMounted] = useState<boolean>(false);

  useEffect(() => {
    setMounted(true);
    // Check local storage or system preference
    const stored = localStorage.getItem("driftguard-theme");
    if (stored === "dark" || (!stored && window.matchMedia("(prefers-color-scheme: dark)").matches)) {
      setIsDark(true);
      document.documentElement.classList.add("dark");
    } else {
      setIsDark(false);
      document.documentElement.classList.remove("dark");
    }
  }, []);

  const toggleTheme = () => {
    const nextDark = !isDark;
    setIsDark(nextDark);
    if (nextDark) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("driftguard-theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("driftguard-theme", "light");
    }
  };

  if (!mounted) {
    return <div className="w-8 h-8 rounded-lg border border-border" />;
  }

  return (
    <button
      onClick={toggleTheme}
      aria-label="Toggle theme"
      className="p-1.5 rounded-lg border border-border hover:bg-surface-hover text-muted hover:text-primary transition-colors flex items-center justify-center"
    >
      {isDark ? <Sun className="h-4 w-4 text-terracotta" /> : <Moon className="h-4 w-4 text-muted" />}
    </button>
  );
}
