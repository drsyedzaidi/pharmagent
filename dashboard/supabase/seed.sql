-- Seed the course catalog. stripe_price_id is a placeholder until the
-- corresponding Stripe Products/Prices exist -- update these 4 rows once
-- you have real price IDs (see SETUP.md).
--
-- Slugs match the directory names under agency-courses/ and the
-- courses/<slug>/ output of build/generate_courses.py -- keep them in sync.

insert into courses (slug, title, price_cents, stripe_price_id, is_active) values
  ('pkpd-modeling-sim',       'PKPD Modeling and Simulation',   14900, null, true),
  ('handling-missing-data-r', 'Handling Missing Data in R',     14900, null, true),
  ('messy-data-r',            'Dealing with Messy Data in R',   14900, null, true),
  ('all-access-bundle',       'All-Access Bundle (3 Courses)',  34900, null, true)
on conflict (slug) do update set
  title = excluded.title,
  price_cents = excluded.price_cents,
  is_active = excluded.is_active;
