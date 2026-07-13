import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { supabase } from "../lib/supabaseClient";
import { callFunction } from "../lib/api";
import { useAuth } from "../lib/AuthContext";
import manifest from "../data/courseManifest.json";

interface Course {
  slug: string;
  title: string;
  price_cents: number;
}

export default function Checkout() {
  const { course: courseSlug } = useParams<{ course: string }>();
  const { user, loading: authLoading } = useAuth();
  const [course, setCourse] = useState<Course | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [redirecting, setRedirecting] = useState(false);

  const manifestEntry = manifest.courses.find((c) => c.slug === courseSlug);

  useEffect(() => {
    if (!courseSlug) return;
    supabase
      .from("courses")
      .select("slug, title, price_cents")
      .eq("slug", courseSlug)
      .eq("is_active", true)
      .maybeSingle()
      .then(({ data, error }) => {
        if (error) setError(error.message);
        else setCourse(data);
      });
  }, [courseSlug]);

  async function handleEnroll() {
    setError(null);
    setRedirecting(true);
    try {
      const { url } = await callFunction<{ url: string }>("create-checkout-session", { course_slug: courseSlug });
      window.location.href = url;
    } catch (e) {
      setRedirecting(false);
      setError(e instanceof Error ? e.message : "Could not start checkout.");
    }
  }

  if (authLoading) return <div className="p-10 text-center text-muted">Loading…</div>;

  if (!user) {
    const next = encodeURIComponent(`/checkout/${courseSlug}`);
    return (
      <div className="mx-auto max-w-md px-6 py-16 text-center">
        <h1 className="text-2xl mb-3">Sign in to enroll</h1>
        <p className="text-muted mb-6">Create an account or sign in first, then come back here to pay.</p>
        <div className="flex justify-center gap-3">
          <Link to={`/login?next=${next}`} className="rounded-lg border border-line-strong px-4 py-2.5 font-medium">
            Sign in
          </Link>
          <Link to={`/signup?next=${next}`} className="rounded-lg bg-primary px-4 py-2.5 text-white font-medium">
            Sign up
          </Link>
        </div>
      </div>
    );
  }

  if (!course) {
    return <div className="p-10 text-center text-muted">{error ? `Course not found: ${error}` : "Loading course…"}</div>;
  }

  const priceLabel = `$${(course.price_cents / 100).toFixed(0)}`;

  return (
    <div className="mx-auto max-w-md px-6 py-16">
      <p className="font-mono text-xs uppercase tracking-wider text-primary mb-3">Enroll</p>
      <h1 className="text-3xl mb-2">{course.title}</h1>
      {manifestEntry && (
        <p className="text-muted mb-6">
          {manifestEntry.subtitle} — {manifestEntry.total_lessons} lessons across {manifestEntry.phases.length} phases.
        </p>
      )}
      <div className="rounded-2xl border border-line bg-surface p-6 shadow-sm">
        <div className="flex items-baseline justify-between mb-6">
          <span className="text-muted">One-time payment</span>
          <span className="text-3xl font-serif font-semibold">{priceLabel}</span>
        </div>
        {error && <p className="text-sm text-danger mb-4">{error}</p>}
        <button
          onClick={handleEnroll}
          disabled={redirecting}
          className="w-full rounded-lg bg-primary px-4 py-3 text-white font-medium hover:bg-primary-dim disabled:opacity-60 cursor-pointer"
        >
          {redirecting ? "Redirecting to checkout…" : `Enroll — ${priceLabel}`}
        </button>
        <p className="text-xs text-faint mt-3 text-center">
          Secure checkout via Stripe. Lifetime access, self-paced.
        </p>
      </div>
    </div>
  );
}
