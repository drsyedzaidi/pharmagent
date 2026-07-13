import { createClient } from "@supabase/supabase-js";

// Client-safe values only: the anon key is designed to be public and relies
// entirely on Postgres RLS to restrict what it can read/write. Never put
// the service-role key here or in any VITE_-prefixed variable -- anything
// prefixed VITE_ gets bundled into the public client JS.
const supabaseUrl = import.meta.env.VITE_SUPABASE_URL as string;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY as string;

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error(
    "Missing VITE_SUPABASE_URL / VITE_SUPABASE_ANON_KEY. Copy .env.example to .env and fill in your Supabase project's values."
  );
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey);
