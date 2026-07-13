import { supabase } from "./supabaseClient";

/** Calls a Netlify Function with the current user's Supabase access token
 * attached as a Bearer header, so the Function can verify identity
 * server-side before doing anything entitlement-sensitive. */
export async function callFunction<T>(name: string, body?: unknown): Promise<T> {
  const { data } = await supabase.auth.getSession();
  const token = data.session?.access_token;
  if (!token) throw new Error("Not signed in.");

  const res = await fetch(`/.netlify/functions/${name}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: body ? JSON.stringify(body) : undefined,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${name} failed (${res.status}): ${text}`);
  }
  return res.json() as Promise<T>;
}
