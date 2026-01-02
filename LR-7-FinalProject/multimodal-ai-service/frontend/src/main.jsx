import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";
import MultimodalAIService from "./MultimodalAIService.jsx";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <MultimodalAIService />
  </StrictMode>
);
