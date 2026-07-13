import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { supabase } from "../lib/supabaseClient";
import { callFunction } from "../lib/api";
import manifest from "../data/courseManifest.json";
import "../styles/lessonContent.css";

const MARKETING_SITE = "https://pharmagent.netlify.app";

interface QuizQuestion {
  id: number;
  q: string;
  options: string[];
  answer: string;
  explanation: string;
}

interface LessonPayload {
  title: string;
  doc_html: string;
  code_sections: { filename: string; html: string }[];
  quiz_questions: QuizQuestion[];
  artifact_html: string;
  images: { name: string; url: string }[];
}

type LoadState = { kind: "loading" } | { kind: "forbidden" } | { kind: "error"; message: string } | { kind: "ready"; data: LessonPayload };

export default function Learn() {
  const { course, phase, lesson } = useParams<{ course: string; phase: string; lesson: string }>();
  const [state, setState] = useState<LoadState>({ kind: "loading" });
  const [marking, setMarking] = useState(false);
  const [marked, setMarked] = useState(false);

  const courseEntry = manifest.courses.find((c) => c.slug === course);
  const flatLessons = courseEntry?.phases.flatMap((p) => p.lessons.map((l) => ({ ...l, phase: p.slug }))) ?? [];
  const currentIndex = flatLessons.findIndex((l) => l.phase === phase && l.slug === lesson);
  const currentEntry = currentIndex >= 0 ? flatLessons[currentIndex] : null;
  const prevEntry = currentIndex > 0 ? flatLessons[currentIndex - 1] : null;
  const nextEntry = currentIndex >= 0 && currentIndex < flatLessons.length - 1 ? flatLessons[currentIndex + 1] : null;

  useEffect(() => {
    if (!course || !phase || !lesson) return;

    // The one free-preview lesson per course never has a Storage-hosted
    // payload (see generate_courses.py) -- it's a public static page on the
    // marketing site. Bounce there instead of calling get-lesson for it.
    if (currentEntry?.free) {
      window.location.href = `${MARKETING_SITE}/courses/${course}/${phase}/${lesson}/index.html`;
      return;
    }

    setState({ kind: "loading" });
    setMarked(false);
    callFunction<LessonPayload>("get-lesson", { course_slug: course, phase_slug: phase, lesson_slug: lesson })
      .then((data) => setState({ kind: "ready", data }))
      .catch((e) => {
        const message = e instanceof Error ? e.message : "Failed to load lesson.";
        if (message.includes("403")) setState({ kind: "forbidden" });
        else setState({ kind: "error", message });
      });
  }, [course, phase, lesson, currentEntry?.free]);

  async function markComplete() {
    if (!course || !phase || !lesson) return;
    setMarking(true);
    const { data: userData } = await supabase.auth.getUser();
    if (!userData.user) return;
    const { error } = await supabase.from("lesson_progress").upsert(
      { user_id: userData.user.id, course_slug: course, phase_slug: phase, lesson_slug: lesson },
      { onConflict: "user_id,course_slug,phase_slug,lesson_slug" }
    );
    setMarking(false);
    if (!error) setMarked(true);
  }

  if (state.kind === "loading") {
    return <div className="p-10 text-center text-muted">Loading lesson…</div>;
  }

  if (state.kind === "forbidden") {
    return (
      <div className="mx-auto max-w-md px-6 py-16 text-center">
        <h1 className="text-2xl mb-3">Enrollment required</h1>
        <p className="text-muted mb-6">
          This lesson is part of <strong>{courseEntry?.title ?? course}</strong>, which you haven't enrolled in yet.
        </p>
        <Link to={`/checkout/${course}`} className="inline-block rounded-lg bg-primary px-4 py-2.5 text-white font-medium">
          Enroll to unlock
        </Link>
      </div>
    );
  }

  if (state.kind === "error") {
    return <div className="p-10 text-center text-danger">{state.message}</div>;
  }

  const { data } = state;

  return (
    <div className="mx-auto max-w-3xl px-6 py-14">
      <nav className="font-mono text-xs text-faint mb-6">
        <Link to="/dashboard" className="text-muted hover:text-primary">
          My Courses
        </Link>{" "}
        / {courseEntry?.title ?? course} / {data.title}
      </nav>

      <div className="lesson-content">
        <article className="lesson-doc" dangerouslySetInnerHTML={{ __html: data.doc_html }} />

        {data.code_sections.length > 0 && (
          <section className="lesson-section">
            <h2>Code</h2>
            {data.code_sections.map((c) => (
              <div key={c.filename}>
                <h3 className="lesson-subhead">{c.filename}</h3>
                <div dangerouslySetInnerHTML={{ __html: c.html }} />
              </div>
            ))}
          </section>
        )}

        {data.images.length > 0 && (
          <section className="lesson-section">
            <h2>Figures</h2>
            <div className="lesson-figures">
              {data.images.map((img) => (
                <figure className="lesson-figure" key={img.name}>
                  <img src={img.url} alt={img.name.replace(/_/g, " ")} loading="lazy" />
                  <figcaption>{img.name}</figcaption>
                </figure>
              ))}
            </div>
          </section>
        )}

        {data.quiz_questions.length > 0 && (
          <section className="lesson-section">
            <h2>Check</h2>
            <div className="quiz-list">
              {data.quiz_questions.map((q) => (
                <details className="quiz-item" key={q.id}>
                  <summary>{q.q}</summary>
                  <ul className="quiz-options">
                    {q.options.map((opt) => (
                      <li key={opt}>{opt}</li>
                    ))}
                  </ul>
                  <p className="quiz-answer">
                    <strong>Answer: {q.answer}</strong>
                  </p>
                  <p className="quiz-explanation">{q.explanation}</p>
                </details>
              ))}
            </div>
          </section>
        )}

        {data.artifact_html && (
          <section className="lesson-section">
            <h2>Ship it</h2>
            <div className="artifact-box" dangerouslySetInnerHTML={{ __html: data.artifact_html }} />
          </section>
        )}
      </div>

      <div className="flex items-center justify-between mt-10 pt-8 border-t border-line">
        <button
          onClick={markComplete}
          disabled={marking || marked}
          className="rounded-lg border border-line-strong px-4 py-2.5 font-medium disabled:opacity-60 cursor-pointer"
        >
          {marked ? "✓ Marked complete" : marking ? "Saving…" : "Mark complete"}
        </button>
        <div className="flex gap-3">
          {prevEntry && (
            <Link
              to={`/learn/${course}/${prevEntry.phase}/${prevEntry.slug}`}
              className="rounded-lg border border-line-strong px-4 py-2.5 font-medium"
            >
              ← Previous
            </Link>
          )}
          {nextEntry && (
            <Link to={`/learn/${course}/${nextEntry.phase}/${nextEntry.slug}`} className="rounded-lg bg-primary px-4 py-2.5 text-white font-medium">
              Next →
            </Link>
          )}
        </div>
      </div>
    </div>
  );
}
