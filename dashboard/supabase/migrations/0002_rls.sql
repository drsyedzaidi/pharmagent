-- Row Level Security.
--
-- Security model:
--   courses          -- readable by anyone (pricing page), writable by nobody
--                        from the client (only the service-role key, used
--                        server-side, bypasses RLS entirely).
--   enrollments      -- a user can SELECT only their own rows. There is
--                        deliberately NO insert/update/delete policy for the
--                        `authenticated` role at all -- the only way a row
--                        gets created is the stripe-webhook Netlify Function,
--                        which uses the service-role key and therefore
--                        bypasses RLS. A client can never grant itself access.
--   lesson_progress  -- a user can select/insert/update only their own rows,
--                        AND only for a course they are actively enrolled in
--                        (enforced by the trigger below, not just the UI) --
--                        otherwise an authenticated-but-unpaid user could
--                        still write progress rows to probe which lessons
--                        exist for a course they haven't bought.

alter table courses enable row level security;
alter table enrollments enable row level security;
alter table lesson_progress enable row level security;

-- courses: public read, no client writes
create policy "courses are publicly readable"
  on courses for select
  using (true);

-- enrollments: read own rows only; no insert/update/delete policy at all
-- for authenticated/anon -- only the service-role key (webhook) can write.
create policy "users read own enrollments"
  on enrollments for select
  to authenticated
  using (auth.uid() = user_id);

-- lesson_progress: read/write own rows only
create policy "users read own lesson progress"
  on lesson_progress for select
  to authenticated
  using (auth.uid() = user_id);

create policy "users insert own lesson progress"
  on lesson_progress for insert
  to authenticated
  with check (auth.uid() = user_id);

create policy "users update own lesson progress"
  on lesson_progress for update
  to authenticated
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

-- Entitlement guard: reject a lesson_progress row unless the user has an
-- active enrollment in that course OR the all-access bundle. Runs with the
-- privileges of the inserting role (the authenticated user), so the existing
-- "users read own enrollments" policy is enough for this SELECT to see the
-- row it needs -- no SECURITY DEFINER required.
create or replace function check_active_enrollment()
returns trigger
language plpgsql
as $$
begin
  if not exists (
    select 1 from enrollments
    where user_id = new.user_id
      and status = 'active'
      and course_slug in (new.course_slug, 'all-access-bundle')
  ) then
    raise exception 'no active enrollment for course %', new.course_slug;
  end if;
  return new;
end;
$$;

drop trigger if exists lesson_progress_requires_enrollment on lesson_progress;
create trigger lesson_progress_requires_enrollment
  before insert or update on lesson_progress
  for each row execute function check_active_enrollment();

-- Convenience RPC so the client (and Netlify Functions, if useful) can ask
-- "am I entitled to this course" in one call instead of hand-rolling the
-- bundle-OR-direct-course logic in every caller. Runs as the calling user
-- (security invoker, the Postgres default) so it only ever sees rows the
-- "users read own enrollments" policy already allows -- no privilege escalation.
create or replace function has_active_enrollment(p_course_slug text)
returns boolean
language sql
stable
as $$
  select exists (
    select 1 from enrollments
    where user_id = auth.uid()
      and status = 'active'
      and course_slug in (p_course_slug, 'all-access-bundle')
  );
$$;
