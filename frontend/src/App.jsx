import { BrowserRouter, Routes, Route } from "react-router-dom";
import Generate from "./pages/Generate";
import Chat from "./pages/Chat";
import ForecastPage from "./pages/ForecastPage";
import Dashboard from "./pages/Dashboard";
import News from "./pages/News"

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* when you go to http://localhost:5173/generate */}
        <Route path="/" element={<Dashboard />} />
        <Route path="/generate" element={<Generate />} />
        <Route path="/chat" element={<Chat />} />
        <Route path="/predict" element={<ForecastPage />} />
        <Route path="/news" element={<News />} />
      </Routes>
    </BrowserRouter>
  );
}
