import { jwtVerify } from "jose";

/** Verifies a Supabase-issued access token locally (no round trip to
 * Supabase) using the project's JWT secret (Settings -> API -> JWT Secret
 * in the Supabase dashboard). Returns the authenticated user's id, or
 * throws if the token is missing/invalid/expired. */
export async function verifyUserToken(authHeader: string | undefined): Promise<string> {
  if (!authHeader?.startsWith("Bearer ")) {
    throw new Error("Missing Authorization header.");
  }
  const token = authHeader.slice("Bearer ".length);
  const secret = process.env.SUPABASE_JWT_SECRET;
  if (!secret) {
    throw new Error("SUPABASE_JWT_SECRET is not set on this Function's environment.");
  }
  const { payload } = await jwtVerify(token, new TextEncoder().encode(secret));
  if (!payload.sub) {
    throw new Error("Token has no subject (user id).");
  }
  return payload.sub;
}
