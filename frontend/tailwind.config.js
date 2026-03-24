/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0b1020",
        panel: "#121a30",
        line: "#2a365a",
        ink: "#eef2ff",
        mute: "#9ea8c7",
        good: "#31d0aa",
        warn: "#ffb86a",
        bad: "#ff6d8a",
        accent: "#69a8ff"
      },
      fontFamily: {
        display: ["Space Grotesk", "ui-sans-serif", "system-ui"],
        body: ["Manrope", "ui-sans-serif", "system-ui"]
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(105,168,255,.25), 0 8px 30px rgba(0,0,0,.35)"
      }
    }
  },
  plugins: []
};

