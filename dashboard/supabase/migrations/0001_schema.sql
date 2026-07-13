-- Course catalog, enrollments, and per-lesson progress.
-- A single "all-access-bundle" course row is treated as an enrollment that
-- satisfies entitlement for every real course (checked in application code
-- and in the trigger below via course_slug IN (real_slug, 'all-access-bundle')).

create extension if not exists "pgcrypto";

create table if not exists courses (
  slug text primary key,
  title text not null,
  price_cents integer not null check (price_cents >= 0),
  stripe_price_id text,
  is_active boolean not null default true
);

create table if not exists enrollments (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  course_slug text not null references courses(slug),
  status text not null check (status in ('active', 'revoked')),
  stripe_checkout_session_id text,
  enrolled_at timestamptz not null default now(),
  unique (user_id, course_slug)
);

create index if not exists enrollments_user_id_idx on enrollments(user_id);

create table if not exists lesson_progress (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  course_slug text not null,
  phase_slug text not null,
  lesson_slug text not null,
  completed_at timestamptz not null default now(),
  unique (user_id, course_slug, phase_slug, lesson_slug)
);

create index if not exists lesson_progress_user_course_idx on lesson_progress(user_id, course_slug);
