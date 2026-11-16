/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: "class",
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        surface: "var(--surface)",
        "surface-hover": "var(--surface-hover)",
        sidebar: "var(--sidebar)",
        border: "var(--border)",
        accent: "var(--primary)",
        terracotta: {
          DEFAULT: "var(--primary)",
          hover: "var(--primary-hover)",
          muted: "var(--primary-muted)",
        },
        primary: "var(--foreground)",
        muted: "var(--muted)",
        success: {
          DEFAULT: "var(--success)",
          muted: "var(--success-muted)",
        },
        warning: {
          DEFAULT: "var(--warning)",
          muted: "var(--warning-muted)",
        },
        danger: {
          DEFAULT: "var(--danger)",
          muted: "var(--danger-muted)",
        },
      },
      borderRadius: {
        xl: "12px",
        "2xl": "14px",
      },
    },
  },
  plugins: [],
};
