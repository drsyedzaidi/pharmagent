import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { supabase } from "../lib/supabaseClient";
import { useAuth } from "../lib/AuthContext";
import manifest from "../data/courseManifest.json";

const MARKETING_SITE = "https://pharmagent.netlify.app";
const BUNDLE_SLUG = "all-access-bundle";

interface CourseCard {
  slug: string;
  title: string;
  subtitle: string;
  totalLessons: number;
  completedLessons: number;
  resume: { phase: string; lesson: string; title: string } | null;
}

export default function Dashboard() {
  const { user } = useAuth();
  const [cards, setCards] = useState<CourseCard[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!user) return;

    async function load() {
      const [{ data: enrollments, error: enrollErr }, { data: progress, error: progressErr }] = await Promise.all([
        supabase.from("enrollments").select("course_slug").eq("status", "active"),
        supabase.from("lesson_progress").select("course_slug, phase_slug, lesson_slug"),
      ]);

      if (enrollErr) return setError(enrollErr.message);
      if (progressErr) return setError(progressErr.message);

      const enrolledSlugs = new Set((enrollments ?? []).map((e) => e.course_slug));
      const hasBundle = enrolledSlugs.has(BUNDLE_SLUG);

      const completedByCourse = new Map<string, Set<string>>();
      for (const p of progress ?? []) {
        const key = `${p.phase_slug}/${p.lesson_slug}`;
        if (!completedByCourse.has(p.course_slug)) completedByCourse.set(p.course_slug, new Set());
        completedByCourse.get(p.course_slug)!.add(key);
      }

      const result: CourseCard[] = [];
      for (const course of manifest.courses) {
        if (!hasBundle && !enrolledSlugs.has(course.slug)) continue;
        const completed = completedByCourse.get(course.slug) ?? new Set<string>();

        let resume: CourseCard["resume"] = null;
        outer: for (const phase of course.phases) {
          for (const lesson of phase.lessons) {
            if (!completed.has(`${phase.slug}/${lesson.slug}`)) {
              resume = { phase: phase.slug, lesson: lesson.slug, title: lesson.title };
              break outer;
            }
          }
        }

        result.push({
          slug: course.slug,
          title: course.title,
          subtitle: course.subtitle,
          totalLessons: course.total_lessons,
          completedLessons: completed.size,
          resume,
        });
      }
      setCards(result);
    }

    load();
  }, [user]);

  return (
    <div className="mx-auto max-w-3xl px-6 py-14">
      <p className="font-mono text-xs uppercase tracking-wider text-primary mb-3">My Courses</p>
      <h1 className="text-3xl mb-8">Welcome back{user?.email ? `, ${user.email}` : ""}.</h1>

      {error && <p className="text-sm text-danger mb-6">{error}</p>}

      {cards === null && <p className="text-muted">Loading your courses…</p>}

      {cards && cards.length === 0 && (
        <div className="rounded-2xl border border-line bg-surface p-8 text-center">
          <p className="text-muted mb-4">You haven't enrolled in a course yet.</p>
          <a href={MARKETING_SITE} className="inline-block rounded-lg bg-primary px-4 py-2.5 text-white font-medium">
            Browse courses
          </a>
        </div>
      )}

      <div className="grid gap-5">
        {cards?.map((c) => {
          const pct = c.totalLessons ? Math.round((c.completedLessons / c.totalLessons) * 100) : 0;
          return (
            <div key={c.slug} className="rounded-2xl border border-line bg-surface p-6">
              <div className="flex items-start justify-between gap-4 mb-3">
                <div>
                  <h2 className="text-xl">{c.title}</h2>
                  <p className="text-sm text-muted">{c.subtitle}</p>
                </div>
                <span className="font-mono text-xs text-muted shrink-0">
                  {c.completedLessons}/{c.totalLessons} lessons
                </span>
              </div>
              <div className="h-2 rounded-full bg-surface-2 overflow-hidden mb-4">
                <div className="h-full bg-primary" style={{ width: `${pct}%` }} />
              </div>
              {c.resume ? (
                <Link
                  to={`/learn/${c.slug}/${c.resume.phase}/${c.resume.lesson}`}
                  className="inline-flex items-center rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white hover:bg-primary-dim"
                >
                  {c.completedLessons === 0 ? "Start" : "Resume"}: {c.resume.title}
                </Link>
              ) : (
                <span className="inline-flex items-center rounded-lg bg-accent-soft px-4 py-2 text-sm font-medium text-success">
                  Course complete
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
