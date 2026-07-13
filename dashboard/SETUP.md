# Setup: going from code to a live, sellable course platform

Everything in this repo is built and ready. What's left needs your own
accounts and credentials — I can't create a Supabase project, a Stripe
account, or enter API keys on your behalf. Follow these steps in order;
each one unblocks the next.

## 1. Create the Supabase project

1. Go to [supabase.com](https://supabase.com), create a new project (free
   tier is enough to start). Pick any region close to you.
2. Once it's provisioned, go to **Project Settings → API** and copy three
   values — you'll need all of them shortly:
   - **Project URL** (`https://<ref>.supabase.co`)
   - **anon public** key
   - **service_role** key (click "reveal" — keep this one secret)
3. Also on that page, copy the **JWT Secret** (under "JWT Settings").
4. Go to the **SQL Editor**, and run these three files in order (paste each
   one's contents and click Run):
   - `supabase/migrations/0001_schema.sql`
   - `supabase/migrations/0002_rls.sql`
   - `supabase/seed.sql`
5. Go to **Storage**, create two buckets:
   - `lesson-content` — set **Private**
   - `lesson-assets` — set **Private**

   (Both stay private; the `get-lesson` Function reads them with the
   service-role key and hands back signed URLs for images, never a public
   bucket URL.)

## 2. Create the Stripe products

1. Go to [stripe.com](https://stripe.com), create an account if you don't
   have one. Stay in **test mode** for now (toggle top-right) — you'll
   switch to live mode only after everything works end to end.
2. Go to **Product catalog → Add product**, create four:
   | Product | Price |
   |---|---|
   | PKPD Modeling and Simulation | $149.00, one-time |
   | Handling Missing Data in R | $149.00, one-time |
   | Dealing with Messy Data in R | $149.00, one-time |
   | All-Access Bundle | $349.00, one-time |
3. For each, copy the **Price ID** (starts `price_...`, not the Product ID).
4. Back in Supabase's SQL Editor, update the `courses` table with the real
   price IDs:
   ```sql
   update courses set stripe_price_id = 'price_...' where slug = 'pkpd-modeling-sim';
   update courses set stripe_price_id = 'price_...' where slug = 'handling-missing-data-r';
   update courses set stripe_price_id = 'price_...' where slug = 'messy-data-r';
   update courses set stripe_price_id = 'price_...' where slug = 'all-access-bundle';
   ```
5. Go to **Developers → API keys**, copy the **Secret key** (`sk_test_...`).
   You'll create the webhook (and get its signing secret) in step 4, after
   the dashboard is deployed and you have a real URL for it to point at.

## 3. Local development

```bash
cd pharmagent/dashboard
cp .env.example .env
# Fill in .env with the Supabase URL/anon key/service-role key/JWT secret
# and the Stripe secret key from steps 1-2. Leave STRIPE_WEBHOOK_SECRET
# blank for now -- you don't have it yet.

npm install   # if you haven't already
npm run dev
```

Visit `http://localhost:5190`, sign up for a test account, and confirm the
"My Courses" page loads (empty, since you haven't enrolled in anything).
Checkout won't fully work yet locally without `netlify dev` (which runs the
Functions) — that's fine, move to step 4 to deploy and test the real thing.

## 4. Deploy the dashboard and wire up the webhook

```bash
cd pharmagent/dashboard
npx --yes netlify-cli login          # if not already logged in
npx --yes netlify-cli deploy --prod --dir=dist --build
```

First run will ask to create a new site — accept, name it (e.g.
`pharmagent-dashboard`), and it'll build (`npm run build`) and deploy. Note
the live URL it gives you.

Now set the real environment variables on this Netlify site (Site settings
→ Environment variables, or via CLI):

```bash
netlify env:set VITE_SUPABASE_URL "https://<ref>.supabase.co"
netlify env:set VITE_SUPABASE_ANON_KEY "..."
netlify env:set SUPABASE_URL "https://<ref>.supabase.co"
netlify env:set SUPABASE_SERVICE_ROLE_KEY "..."
netlify env:set SUPABASE_JWT_SECRET "..."
netlify env:set STRIPE_SECRET_KEY "sk_test_..."
```

Then, in Stripe: **Developers → Webhooks → Add endpoint**, URL =
`https://<your-dashboard-url>/.netlify/functions/stripe-webhook`, select
event `checkout.session.completed`. Copy its **Signing secret**
(`whsec_...`) and set it too:

```bash
netlify env:set STRIPE_WEBHOOK_SECRET "whsec_..."
```

Redeploy once more so the Functions pick up the new env vars:

```bash
npx --yes netlify-cli deploy --prod --dir=dist --build
```

## 5. Point the marketing site's links at the real dashboard URL

If your dashboard's live URL isn't exactly `pharmagent-dashboard.netlify.app`,
update the one constant it's hardcoded from:

- `pharmagent/website/build/generate_courses.py` → `DASHBOARD_BASE_URL`

Then regenerate and redeploy the marketing site:

```bash
cd pharmagent/website
build/siteenv/bin/python build/generate_courses.py
npx --yes netlify-cli deploy --prod --dir=.
```

(The course cards' "Enroll" links in `courses.html` also hardcode this URL
— update those three `https://pharmagent-dashboard.netlify.app/checkout/...`
hrefs plus the bundle banner's if you changed the domain.)

## 6. Upload the gated lesson content

```bash
cd pharmagent/website
export SUPABASE_URL="https://<ref>.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="..."   # same service-role key as above
build/siteenv/bin/pip install supabase    # one-time
build/siteenv/bin/python build/upload_lesson_content.py
```

This pushes all 62 gated lessons' JSON + images to Supabase Storage. Rerun
this (and the generator before it) any time lesson content changes — no
app redeploy needed for content-only updates.

## 7. End-to-end test (Stripe test mode)

1. Visit the marketing site's pricing page, click **Enroll** on any course.
2. Sign up for an account (or sign in).
3. You'll land on Stripe's hosted checkout. Pay with Stripe's test card:
   `4242 4242 4242 4242`, any future expiry, any CVC, any ZIP.
4. You should land back on `/dashboard?checkout=success` and see the course
   listed as enrolled (may take a couple seconds for the webhook to land —
   refresh if it's not there instantly).
5. Click into the course, confirm the lesson content, code highlighting,
   quiz, and images all render.
6. Click **Mark complete** on a lesson, go back to the dashboard, confirm
   the progress bar updated and "Resume" points at the next lesson.
7. Try visiting `/learn/<a-course-you-didn't-buy>/...` directly — confirm
   you get the "Enrollment required" paywall, not the content.

Once all of that works in test mode, flip Stripe to live mode (new live
secret key + a live-mode webhook endpoint with its own signing secret —
repeat the relevant env var updates), and you're actually charging real
money.
