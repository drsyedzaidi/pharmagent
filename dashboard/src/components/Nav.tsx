import { Link } from "react-router-dom";
import { useAuth } from "../lib/AuthContext";
import { supabase } from "../lib/supabaseClient";

const MARKETING_SITE = "https://pharmagent.netlify.app";

export function Nav() {
  const { user } = useAuth();

  return (
    <header className="sticky top-0 z-50 bg-base/90 backdrop-blur border-b border-line">
      <div className="mx-auto max-w-5xl px-6 h-16 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2.5 text-ink">
          <span className="grid place-items-center w-8 h-8 rounded-lg bg-primary text-white font-serif text-lg font-semibold">℞</span>
          <span className="font-serif text-lg font-semibold">PharmAgent</span>
        </Link>
        <nav className="flex items-center gap-6 text-sm font-medium text-muted">
          <a href={MARKETING_SITE}>Courses</a>
          {user && <Link to="/dashboard">My Courses</Link>}
        </nav>
        <div>
          {user ? (
            <button
              onClick={() => supabase.auth.signOut()}
              className="text-sm font-medium text-muted hover:text-ink cursor-pointer"
            >
              Sign out
            </button>
          ) : (
            <Link
              to="/login"
              className="inline-flex items-center rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-dim"
            >
              Sign in
            </Link>
          )}
        </div>
      </div>
    </header>
  );
}
