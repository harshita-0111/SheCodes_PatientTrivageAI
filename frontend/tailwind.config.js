/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        triage: {
          critical: "#dc2626",
          urgent: "#ea580c",
          moderate: "#ca8a04",
          low: "#16a34a",
          nonurgent: "#2563eb",
        },
        surface: {
          bg: "#F4F6F8",      // Light grey background
          panel: "#FFFFFF",   // White cards/panels
          border: "#ECEFF3",  // Light border color
          muted: "#9AA1A9",   // Muted grey text
          ink: "#1A1D1F",     // Dark ink text for light background
        },
        accent: {
          mint: "#34D07F",    // Mint accent green
          mintInk: "#1B9A63", // Mint green text
          wash: "#F1F8F4",    // Soft green wash background
          blue: "#1E40AF",    // Deep blue primary button color
          alert: "#FF5A5A",
        },
      },
      boxShadow: {
        card: "0 8px 30px -12px rgba(16,24,40,0.10)",
        lift: "0 18px 40px -16px rgba(16,24,40,0.18)",
      },
      borderRadius: { panel: "14px" }, // Maintain: 14px border radius
    },
  },
  plugins: [],
};
