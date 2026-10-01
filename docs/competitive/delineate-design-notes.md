# Design notes from delineate.pro, and what they mean for the PharmAgent site

Source: homepage screenshot and computed styles, read 2026-10-01. These are observations to learn from, not a design to copy. Their visual identity (dark glow, floating document illustration) is theirs.

## What they do

- **Ground and type.** Near-black navy page, white headings, muted grey-blue body text. Plus Jakarta Sans for display, Inter for body. Periwinkle pill buttons with dark text, 10px radius.
- **Hero.** Left: three-line headline in descending weight with a blue accent third line, one paragraph, two buttons (Request a Demonstration, How It Works). Right: an illustration of source documents (paper, chart, trial, FDA guidance, KM curve) feeding a structured data table. The picture explains the product in one glance.
- **Credibility block.** A single large "4 of the top 10 pharma companies" stat, then a strip of funder logos.
- **Capability explorer.** A "What we do" panel with six vertical tabs on the left and an animated diagram plus paragraph on the right. One panel carries six ideas without six sections.
- **Case studies.** Tabbed. Each opens with a pill diagram of the engagement (for example "300 Trials extracted, Harmonized arm-level, NONMEM-ready, Go"), then the story with concrete numbers.
- **Conversion.** One demo-request card at the end: email, description of your asset, "We'll respond within 24 hours."
- **Trust detail.** A "How we do it" list: dual independent extraction, source-level verification, outlier flagging with rationale, full audit trail.

## What does not work on their page

- In the static render, "Who we support" and "How we do it" show as large empty areas. Content appears only after scroll animations fire, so anything that blocks the reveal leaves a blank page. Do not gate content on motion.
- Body copy is small, low-contrast grey on near-black, likely under the 4.5:1 AA threshold (contrast not measured).
- The hero heading is 32px; the visual weight sits on the illustration, not the claim.
- The "4 of the top 10" claim is unattributed.

## Principles worth borrowing

1. **Show the product's real artefact in the hero.** They draw documents becoming a dataset. PharmAgent should show the Workbench itself: the review panel with its QC checklist and audit trail. It is the most distinctive thing the product has, and it is real.
2. **One tabbed explorer instead of a feature list.** Capabilities on the left, a real screenshot on the right: NCA, population PK, diagnostics, review gate, audit trail, exports.
3. **Case studies as an input, process, decision strip.** For PharmAgent: dataset in, agent steps and human gate, report out. Use real runs (the 12-subject theophylline NCA reached its gate in under a second; the 16-subject population fit took 180 seconds).
4. **One conversion, with a promised response time.** Replace the generic contact with a walkthrough request that says what the visitor gets and when.
5. **A "how we keep it honest" list.** PharmAgent can state it more strongly: every number comes from a deterministic tool, every call is sealed, every gate is a person.

## What to keep different

- **Light, clinical canvas, not dark glow.** The category leans dark and luminous. PharmAgent's cool paper background (`#ECF1F7`) with a single blue accent already reads differently and prints better in a regulatory setting.
- **Evidence over claims.** Use the repo's own verified numbers (the benchmark in the README, the NONMEM reference check) rather than unattributed customer counts. Do not publish customer or funding claims that cannot be shown.
- **Accessible by default.** Keep body text at AA or better, and render all content without scroll animation.

## Concrete changes to the PharmAgent site (not yet made)

1. Hero: keep the current headline, replace the abstract illustration with a real Workbench screenshot of the NCA run at its review gate.
2. Add a tabbed capability explorer (six tabs, each with a real screenshot).
3. Replace the "How it works" strip with an input, process, decision flow using the real run numbers above.
4. Add a "Proof" band with the benchmark result and the NONMEM comparison, each linked to its source.
5. Align type with the app: Spectral for headings can stay for brand character, but body and UI text should match the app's Public Sans.
6. Single walkthrough request form with a stated response time.
