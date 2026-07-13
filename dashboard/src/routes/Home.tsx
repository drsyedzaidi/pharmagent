import { Navigate } from "react-router-dom";
import { useAuth } from "../lib/AuthContext";

const MARKETING_SITE = "https://pharmagent.netlify.app/courses.html";

export default function Home() {
  const { user, loading } = useAuth();

  if (loading) return <div className="p-10 text-center text-muted">Loading…</div>;
  if (user) return <Navigate to="/dashboard" replace />;

  // Unauthenticated visitors landing here directly (rather than via a
  // /checkout/:course or /learn/... deep link) almost certainly meant to
  // browse the catalog on the marketing site, not this app.
  window.location.href = MARKETING_SITE;
  return null;
}
