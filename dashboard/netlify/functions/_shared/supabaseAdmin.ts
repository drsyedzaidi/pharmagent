import { createClient } from "@supabase/supabase-js";

/** Service-role client. Bypasses RLS entirely -- only ever import this in
 * Netlify Functions (server-side), never in src/. The service-role key must
 * be set as a plain (non VITE_-prefixed) environment variable on the
 * Netlify site so it never reaches the client bundle. */
export function supabaseAdmin() {
  const url = process.env.SUPABASE_URL;
  const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !serviceKey) {
    throw new Error("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set on this Function's environment.");
  }
  return createClient(url, serviceKey, { auth: { persistSession: false } });
}

const BUNDLE_SLUG = "all-access-bundle";

/** Mirrors the has_active_enrollment SQL function / lesson_progress trigger
 * logic: a user is entitled to a course if they have an active enrollment
 * in that specific course OR in the all-access bundle. */
export async function hasActiveEnrollment(userId: string, courseSlug: string): Promise<boolean> {
  const { data, error } = await supabaseAdmin()
    .from("enrollments")
    .select("course_slug")
    .eq("user_id", userId)
    .eq("status", "active")
    .in("course_slug", [courseSlug, BUNDLE_SLUG]);

  if (error) throw error;
  return (data?.length ?? 0) > 0;
}
