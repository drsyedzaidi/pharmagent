# Deploying the PharmAgent site to Netlify

Static site (`index.html` + `styles.css` + `netlify.toml`). No build step.
Node 24 and npm 11 are already installed on this machine, so no global install
is needed — `npx` fetches the Netlify CLI on demand.

## 1. Deploy (two commands, one interactive login)

From this folder:

```bash
cd ~/Desktop/Bupropion/pharmagent/website

# Opens a browser once to authorize the Netlify CLI with your account.
npx netlify-cli login

# Create the site + push to production in one go.
# --dir=. publishes this folder; picks up netlify.toml headers automatically.
npx netlify-cli deploy --prod --dir=. --site=pharmagent
```

First run will prompt to create a new site — accept, and it returns a live URL
like `https://pharmagent.netlify.app`. Re-run the `deploy` line any time to ship
an update (login is remembered).

> Prefer no CLI? Netlify Drop is a zero-account-friction alternative: open
> <https://app.netlify.com/drop> and drag the `website/` folder onto the page.
> (You still need to log in to attach a custom domain later.)

## 2. Buy a domain

You don't have one yet. Recommended registrars (cheap, clean, no upsell spam):

- **Cloudflare Registrar** — at-cost pricing, best if you'll use Cloudflare DNS.
- **Porkbun** or **Namecheap** — simple, good `.ai` / `.com` pricing.

Name ideas (check availability):

| Domain | Note |
|--------|------|
| `pmatricsai.com` | Matches the PmatricsAI brand — first choice. |
| `pmatrics.ai` | `.ai` reads well for an AI product (pricier, ~$70–100/yr). |
| `pharmagent.ai` | Product-name-forward. |
| `getpharmagent.com` | Fallback if `.com` is taken. |

## 3. Point the domain at Netlify

After the site is live, in the Netlify dashboard: **Site → Domain management →
Add a custom domain**, enter your domain, then set DNS at your registrar.

**Option A — Netlify DNS (simplest):** Netlify gives you 4 nameservers
(e.g. `dns1.p0X.nrt.netlify.com`). Paste them into your registrar's nameserver
settings. Netlify then manages records and provisions HTTPS automatically.

**Option B — Keep external DNS:** add these records at your registrar:

| Type | Host | Value |
|------|------|-------|
| `A` (or ALIAS/ANAME) | `@` (apex) | `75.2.60.5` |
| `CNAME` | `www` | `<your-site>.netlify.app` |

Netlify auto-issues a free Let's Encrypt certificate once DNS resolves
(usually minutes, up to a few hours for propagation). Force HTTPS is on by
default — the `Strict-Transport-Security` header is already set in
`netlify.toml`.

## Notes

- Security headers (CSP, HSTS, X-Frame-Options, etc.) are defined in
  `netlify.toml` and applied by Netlify's edge. The page ships **zero
  JavaScript**, so the CSP sets `script-src 'none'`.
- To bust the CSS cache after edits, rename `styles.css` (e.g. `styles.v2.css`)
  and update the `<link>` in `index.html`.
