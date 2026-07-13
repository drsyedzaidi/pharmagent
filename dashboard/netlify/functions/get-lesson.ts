import { verifyUserToken } from "./_shared/verifyJwt";
import { hasActiveEnrollment, supabaseAdmin } from "./_shared/supabaseAdmin";

const SIGNED_URL_TTL_SECONDS = 3600;

export default async (req: Request) => {
  if (req.method !== "POST") return new Response("Method not allowed", { status: 405 });

  let userId: string;
  try {
    userId = await verifyUserToken(req.headers.get("authorization") ?? undefined);
  } catch (e) {
    return new Response(e instanceof Error ? e.message : "Unauthorized", { status: 401 });
  }

  const body = (await req.json().catch(() => null)) as
    | { course_slug?: string; phase_slug?: string; lesson_slug?: string }
    | null;
  const courseSlug = body?.course_slug;
  const phaseSlug = body?.phase_slug;
  const lessonSlug = body?.lesson_slug;
  if (!courseSlug || !phaseSlug || !lessonSlug) {
    return new Response("Missing course_slug/phase_slug/lesson_slug", { status: 400 });
  }

  const entitled = await hasActiveEnrollment(userId, courseSlug);
  if (!entitled) {
    return new Response("Not enrolled in this course.", { status: 403 });
  }

  const admin = supabaseAdmin();
  const contentPath = `${courseSlug}/${phaseSlug}/${lessonSlug}.json`;
  const { data: file, error: downloadError } = await admin.storage.from("lesson-content").download(contentPath);
  if (downloadError || !file) {
    return new Response("Lesson content not found.", { status: 404 });
  }

  const payload = JSON.parse(await file.text());

  const images: { name: string; url: string }[] = [];
  for (const imageName of payload.images ?? []) {
    const assetPath = `${courseSlug}/${phaseSlug}/${lessonSlug}/${imageName}`;
    const { data: signed } = await admin.storage.from("lesson-assets").createSignedUrl(assetPath, SIGNED_URL_TTL_SECONDS);
    if (signed?.signedUrl) images.push({ name: imageName, url: signed.signedUrl });
  }

  return Response.json({
    title: payload.title,
    doc_html: payload.doc_html,
    code_sections: payload.code_sections,
    quiz_questions: payload.quiz_questions,
    artifact_html: payload.artifact_html,
    images,
  });
};
