import React from "react";
import { createRoot } from "react-dom/client";
import { Toaster } from "sonner";
import App from "./App";
import "./index.css";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
    <Toaster
      richColors
      position="top-right"
      expand={false}
      toastOptions={{
        style: {
          background: "#111622",
          color: "#E2E8F0",
          border: "1px solid #2A3650",
          fontFamily: "Inter, system-ui, sans-serif",
        },
      }}
    />
  </React.StrictMode>
);
