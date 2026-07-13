import { Routes, Route } from "react-router-dom";
import { Nav } from "./components/Nav";
import { ProtectedRoute } from "./components/ProtectedRoute";
import Home from "./routes/Home";
import Login from "./routes/Login";
import Signup from "./routes/Signup";
import Checkout from "./routes/Checkout";
import Dashboard from "./routes/Dashboard";
import Learn from "./routes/Learn";

export default function App() {
  return (
    <>
      <Nav />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/checkout/:course" element={<Checkout />} />
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          }
        />
        <Route
          path="/learn/:course/:phase/:lesson"
          element={
            <ProtectedRoute>
              <Learn />
            </ProtectedRoute>
          }
        />
      </Routes>
    </>
  );
}
