/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        dark: {
          950: '#06080C',
          900: '#0B0E14',
          800: '#111622',
          700: '#1A2033',
          600: '#232B3E',
          500: '#2C3650',
          400: '#3D4A64',
        },
        accent: {
          DEFAULT: '#00D4AA',
          light: '#33DFBE',
          dark: '#00B894',
          glow: 'rgba(0,212,170,0.12)',
        },
        profit: '#0ECB81',
        loss: '#F6465D',
        warn: '#FCD535',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      boxShadow: {
        card: '0 2px 8px rgba(0,0,0,0.3)',
        glow: '0 0 24px rgba(0,212,170,0.12)',
      },
      animation: {
        'fade-in': 'fadeIn 0.25s ease-out',
        'slide-up': 'slideUp 0.3s ease-out',
      },
      keyframes: {
        fadeIn: { '0%': { opacity: '0' }, '100%': { opacity: '1' } },
        slideUp: { '0%': { opacity: '0', transform: 'translateY(6px)' }, '100%': { opacity: '1', transform: 'translateY(0)' } },
      },
    },
  },
  plugins: [],
};
