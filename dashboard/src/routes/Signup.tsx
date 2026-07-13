import { useState, type FormEvent } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { supabase } from "../lib/supabaseClient";

export default function Signup() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [checkEmail, setCheckEmail] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const next = params.get("next") || "/dashboard";

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    const { data, error } = await supabase.auth.signUp({ email, password });
    setSubmitting(false);
    if (error) {
      setError(error.message);
      return;
    }
    // If email confirmation is required, Supabase returns a user but no
    // session yet -- show a "check your email" state instead of navigating.
    if (data.session) {
      navigate(next, { replace: true });
    } else {
      setCheckEmail(true);
    }
  }

  if (checkEmail) {
    return (
      <div className="mx-auto max-w-sm px-6 py-16 text-center">
        <h1 className="text-2xl mb-3">Check your email</h1>
        <p className="text-muted">
          We sent a confirmation link to <strong className="text-ink">{email}</strong>. Click it, then come
          back and sign in.
        </p>
        <Link to="/login" className="inline-block mt-6 text-primary font-medium">
          Go to sign in
        </Link>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-sm px-6 py-16">
      <p className="font-mono text-xs uppercase tracking-wider text-primary mb-3">Create account</p>
      <h1 className="text-3xl mb-6">Start learning.</h1>
      <form onSubmit={handleSubmit} className="grid gap-4">
        <label className="grid gap-1.5 text-sm text-muted">
          Email
          <input
            type="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="rounded-lg border border-line px-3.5 py-2.5 text-ink bg-surface focus:outline-none focus:border-primary"
          />
        </label>
        <label className="grid gap-1.5 text-sm text-muted">
          Password
          <input
            type="password"
            required
            minLength={8}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="rounded-lg border border-line px-3.5 py-2.5 text-ink bg-surface focus:outline-none focus:border-primary"
          />
          <span className="text-xs text-faint">At least 8 characters.</span>
        </label>
        {error && <p className="text-sm text-danger">{error}</p>}
        <button
          type="submit"
          disabled={submitting}
          className="rounded-lg bg-primary px-4 py-2.5 text-white font-medium hover:bg-primary-dim disabled:opacity-60 cursor-pointer"
        >
          {submitting ? "Creating account…" : "Create account"}
        </button>
      </form>
      <p className="mt-6 text-sm text-muted">
        Already have an account?{" "}
        <Link to={`/login?next=${encodeURIComponent(next)}`} className="text-primary font-medium">
          Sign in
        </Link>
      </p>
    </div>
  );
}
