---
version: alpha
name: Clinical-Blue-design-system
description: "The house design language for a clinical pharmacology and pharmacometrics consulting agency. A calm, scientific, editorial-clinical system built on a cool paper canvas (#ECF1F7), a single confident clinical blue (#1F66A6) used semantically — for primary actions, focus, links, and the brand mark — never decoratively, and a deep cool ink (#16314A). Display type is set in Spectral (a characterful serif) at 500-600 with mild negative tracking, signalling editorial authority; body is Public Sans (deliberately NOT Inter/Roboto); quantitative values, parameters, and equations are set in IBM Plex Mono to read as data. Surfaces are white panels with hairline cool borders and minimal, flat depth — credibility over decoration. The brand mark is a serif prescription monogram (℞). No gradients, no purple, no generic SaaS slop."

colors:
  primary: "#1F66A6"
  on-primary: "#FFFFFF"
  primary-hover: "#17537F"
  primary-active: "#123F61"
  primary-focus: "#1F66A6"
  accent-soft: "#E3EEF9"
  accent-soft-border: "#B5D4F4"
  ink: "#16314A"
  ink-muted: "#5E7388"
  ink-subtle: "#8298AC"
  ink-on-accent: "#0C447C"
  canvas: "#ECF1F7"
  surface-1: "#FFFFFF"
  surface-2: "#F4F7FB"
  surface-3: "#EAF0F7"
  hairline: "#D7E2EE"
  hairline-strong: "#C5D4E4"
  semantic-success: "#1D7A5A"
  semantic-warning: "#9A5B12"
  semantic-danger: "#B23A2E"
  semantic-info: "#1F66A6"

typography:
  display-xl:
    fontFamily: Spectral
    fontSize: 60px
    fontWeight: 600
    lineHeight: 1.05
    letterSpacing: -1.5px
  display-lg:
    fontFamily: Spectral
    fontSize: 42px
    fontWeight: 600
    lineHeight: 1.10
    letterSpacing: -1.0px
  display-md:
    fontFamily: Spectral
    fontSize: 30px
    fontWeight: 600
    lineHeight: 1.18
    letterSpacing: -0.4px
  headline:
    fontFamily: Spectral
    fontSize: 22px
    fontWeight: 600
    lineHeight: 1.25
    letterSpacing: -0.2px
  card-title:
    fontFamily: Spectral
    fontSize: 18px
    fontWeight: 600
    lineHeight: 1.30
    letterSpacing: 0px
  body-lg:
    fontFamily: Public Sans
    fontSize: 18px
    fontWeight: 400
    lineHeight: 1.65
  body:
    fontFamily: Public Sans
    fontSize: 16px
    fontWeight: 400
    lineHeight: 1.65
  small:
    fontFamily: Public Sans
    fontSize: 14px
    fontWeight: 400
    lineHeight: 1.55
  caption:
    fontFamily: Public Sans
    fontSize: 12.5px
    fontWeight: 500
    lineHeight: 1.45
  data:
    fontFamily: IBM Plex Mono
    fontSize: 14px
    fontWeight: 400
    lineHeight: 1.5
---

## Overview

This is the brand system for a clinical-pharmacology and pharmacometrics consulting agency (PBPK / popPK modelling, model-informed drug development, and the tools and courses around it). The personality is **trustworthy, precise, and quietly premium** — a scientist's notebook crossed with an editorial journal, not a startup landing page. Every surface should read as if a careful modeller designed it: clear hierarchy, honest data, no noise.

Three moves define the look:
1. **One clinical blue, used with restraint** — `#1F66A6` carries primary actions, links, focus rings, and the brand mark. It never appears as a gradient or a decorative wash.
2. **Editorial serif headlines over a cool paper canvas** — Spectral display type gives authority; the cool off-white `#ECF1F7` canvas reads clinical, not stark.
3. **Numbers are data** — parameters, clearances, p-values, doses are set in IBM Plex Mono so quantitative content is legible and credible.

## Colors

### Brand & Accent
- **Primary** `#1F66A6` — buttons, links, active nav, focus rings, the ℞ mark. Text on it is white (`#FFFFFF`).
- **Hover / Active** `#17537F` / `#123F61`.
- **Accent soft** `#E3EEF9` (fill) with `#B5D4F4` border and `#0C447C` text — for pills, tags, highlighted rows, selected states.

### Surface
- **Canvas** `#ECF1F7` — the page background (cool paper).
- **Surface-1** `#FFFFFF` — cards, panels, inputs-at-rest-on-canvas.
- **Surface-2** `#F4F7FB` — insets, code/data blocks, secondary fills.
- **Hairline** `#D7E2EE` (default borders), **hairline-strong** `#C5D4E4` (hover/emphasis).

### Text
- **Ink** `#16314A` — headings and primary text (deep cool navy, not pure black).
- **Ink-muted** `#5E7388` — secondary text, labels, captions.
- **Ink-subtle** `#8298AC` — hints, disabled, metadata.
- On colored fills, use the family's darkest shade (e.g. `#0C447C` on `#E3EEF9`) — never plain black or gray.

### Semantic
- Success `#1D7A5A` · Warning `#9A5B12` · Danger `#B23A2E` · Info `#1F66A6`. Use each only for its meaning (validation pass, caution, error). Reserve red for genuine errors.

## Typography

### Font Family
- **Display:** Spectral (serif) — headlines, the wordmark, section labels. Weights 500-600.
- **Body:** Public Sans — all running text, UI labels, controls. **Never Inter, Roboto, Arial, or a system default** — those read as generic AI output.
- **Data/Mono:** IBM Plex Mono — parameter values, equations, model output, code, doses, statistics.

### Hierarchy
Use the scale in the frontmatter (`display-xl` 60px → `caption` 12.5px). Lead pages with `display-lg`/`display-xl` Spectral; section heads in `headline`; card titles in `card-title`; body at 16px with 1.65 line-height for readability.

### Principles
- Two body weights only: 400 regular, 500 for emphasis/labels. Avoid 700 on body.
- Sentence case everywhere. Never ALL CAPS, never Title Case headings.
- Set every quantitative value (CL, Vd, AUC, Cmax, doses, CIs, p-values) in `data` mono — it visually separates evidence from prose.

### Note on Font Substitutes
If Spectral is unavailable, fall back to Georgia, then a serif stack. If Public Sans is unavailable, fall back to "IBM Plex Sans", then system-ui — but install Public Sans for production. IBM Plex Mono falls back to ui-monospace.

## Layout

### Spacing System
4px base scale: 4, 8, 12, 16, 24, 32, 48, 64, 96. Component-internal gaps use 8/12/16; section rhythm uses 48/64/96. Vary rhythm intentionally — not uniform padding everywhere.

### Grid & Container
Centered content column, max-width 1120px. 12-column grid for marketing surfaces; generous gutters. Data-dense app views may go full-width with a left rail.

### Whitespace Philosophy
Whitespace signals confidence. Let headlines and key numbers breathe. Crowding reads as a dashboard-by-numbers; this brand is editorial-clinical.

## Elevation & Depth

Flat and honest. Depth comes from hairline borders and surface contrast, not heavy shadows. Allowed: a single soft shadow on raised modals/popovers (`0 8px 24px rgba(22,49,74,0.08)`). No neon, no glow, no glassmorphism unless a specific surface justifies it.

## Shapes

### Border Radius Scale
- sm `8px` (inputs, small controls, pills-inline)
- md `12px` (buttons, cards-compact)
- lg `14px` (cards, panels)
- xl `18px` (hero panels, the brand header)
- pill `999px` (tags, status chips)
No rounded corners on single-sided borders (e.g. a `border-left` accent stays square).

## Components

### Buttons
- **Primary:** fill `#1F66A6`, text white, radius `md`, weight 500; hover `#17537F`, active `#123F61` + `scale(0.98)`; focus ring `0 0 0 3px rgba(31,102,166,0.35)`.
- **Secondary:** white fill, `#C5D4E4` border, ink text; hover fill `#ECF1F7`.
- **Ghost:** transparent, ink-muted text, hover surface-2.

### Cards & Containers
White (`surface-1`), `hairline` border, radius `lg`, padding 16-24px. Hover lifts the border to `#1F66A6` for interactive cards. Section labels above cards in Spectral `headline`, ink-muted.

### Inputs & Forms
Fill `surface-2` (`#F4F7FB`), `#C5D4E4` border, radius `sm`, 40px tall; focus border `#1F66A6` + 3px focus ring. Labels in `caption`, ink-muted. Validation uses semantic colors with an icon, never color alone.

### Pills / Tags / Status
`accent-soft` fill, `accent-soft-border`, `#0C447C` text, `pill` radius, 12.5px. Semantic variants swap to the matching semantic family.

### Data & Tables
Quantitative cells in IBM Plex Mono. Header row in `caption` ink-muted; hairline row separators; selected/hover row tinted `surface-3`. Right-align numbers.

### Navigation
Top bar on `surface-1` with a hairline bottom border; the ℞ mark in a `#1F66A6` tile + Spectral wordmark; active item in `#1F66A6` with a 2px underline.

### Brand Mark
Serif `℞` (U+211E) in white on a `#1F66A6` rounded tile (radius 13/46 ≈ 28%), paired with a Spectral wordmark. App icon = the same ℞ on a clinical-blue squircle.

## Do's and Don'ts

### Do
- Use one clinical blue, semantically (actions, links, focus, mark).
- Lead with Spectral serif headlines and real numbers in mono.
- Keep surfaces flat, bordered, and calm; let whitespace carry confidence.
- Treat data visualisation as part of the system (mono labels, the blue accent, hairline grids).
- Make hover/focus/active states feel deliberate.

### Don't
- No purple/blue gradients, mesh, noise, or glow.
- No Inter / Roboto / Arial / system-default body type.
- No uniform card grids with no hierarchy; no dashboard-by-numbers.
- Don't use blue decoratively or in more than ~10% of any view.
- Don't copy another brand's look — this system is the agency's own.
