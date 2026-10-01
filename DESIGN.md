---
name: PharmAgent
description: Audited PK/PD analysis workspace where agents plan, deterministic tools compute and a person approves each gate.
colors:
  cool-paper: "#EEF2F7"
  surface: "#FFFFFF"
  aside: "#F6F8FB"
  border: "#E1E8F0"
  border-subtle: "#EDF1F6"
  border-strong: "#CFD9E4"
  ink: "#2F4557"
  ink-heading: "#16314A"
  ink-dim: "#536A83"
  clinical-blue: "#1F66A6"
  clinical-blue-deep: "#17537F"
  clinical-blue-wash: "#E8F0F9"
  chip-wash: "#F1F5F9"
  verdict-green: "#1D7A5A"
  verdict-green-wash: "#E7F3EE"
  amber: "#9A5B12"
  amber-wash: "#FBF1E4"
  verdict-red: "#B23A2E"
  verdict-red-wash: "#F9E9E7"
typography:
  headline: { fontFamily: "Public Sans, -apple-system, Segoe UI, sans-serif", fontSize: "22px", fontWeight: 600, letterSpacing: "-0.01em" }
  title: { fontFamily: "Public Sans, sans-serif", fontSize: "15px", fontWeight: 600 }
  body: { fontFamily: "Public Sans, sans-serif", fontSize: "13px", fontWeight: 400, lineHeight: 1.45, fontFeature: "tnum" }
  label: { fontFamily: "Public Sans, sans-serif", fontSize: "12px", fontWeight: 600 }
  stat: { fontFamily: "Public Sans, sans-serif", fontSize: "18px", fontWeight: 600 }
  mono: { fontFamily: "IBM Plex Mono, ui-monospace, monospace", fontSize: "11px", fontWeight: 400 }
rounded: { xs: "4px", sm: "6px", md: "8px", lg: "10px", pill: "12px" }
spacing: { xs: "4px", sm: "8px", md: "12px", lg: "16px", xl: "20px", 2xl: "28px", 3xl: "40px" }
components:
  button-primary: { backgroundColor: "{colors.clinical-blue}", textColor: "{colors.surface}", rounded: "{rounded.sm}", padding: "0 12px", height: "32px" }
  button-primary-hover: { backgroundColor: "{colors.clinical-blue-deep}" }
  button-ghost: { backgroundColor: "{colors.surface}", textColor: "{colors.ink-heading}", rounded: "{rounded.sm}", padding: "0 12px", height: "32px" }
  button-ghost-hover: { backgroundColor: "{colors.aside}" }
  button-reject: { backgroundColor: "{colors.surface}", textColor: "{colors.verdict-red}", rounded: "{rounded.sm}", height: "36px" }
  button-reject-hover: { backgroundColor: "{colors.verdict-red-wash}" }
  button-disabled: { backgroundColor: "{colors.aside}", textColor: "{colors.ink-dim}" }
  chip: { backgroundColor: "{colors.surface}", textColor: "{colors.ink}", rounded: "{rounded.sm}", padding: "0 10px", height: "28px" }
  chip-pressed: { backgroundColor: "{colors.clinical-blue-wash}", textColor: "{colors.ink-heading}" }
  input: { backgroundColor: "{colors.surface}", textColor: "{colors.ink-heading}", rounded: "{rounded.sm}", padding: "8px 10px" }
  card: { backgroundColor: "{colors.surface}", rounded: "{rounded.lg}", padding: "14px" }
  status-pill-pending: { backgroundColor: "{colors.amber-wash}", textColor: "{colors.amber}", rounded: "{rounded.pill}", padding: "3px 9px" }
  id-chip: { backgroundColor: "{colors.chip-wash}", textColor: "{colors.ink-dim}", typography: "{typography.mono}", rounded: "{rounded.xs}", padding: "1px 6px" }
---

# Design System: PharmAgent

## Overview

**Creative North Star: "The Sealed Analysis Document"**

A light, cool-paper workspace in three columns: a rail for dataset and step progress, a centre document where each completed tool step is a section of the analysis, and a review column where the human decision lives beside its QC evidence. Chat is a composer under the document, not the stage. The system is flat, quiet and dense where numbers are, roomy everywhere else; one blue carries every call to action, and verdict colours only ever appear with an icon and a word.

**Key Characteristics:**
- Base 13px Public Sans, tabular numerals on by default; IBM Plex Mono for IDs, tool names, hashes, timings.
- Tonal layering (cool-paper / aside / white) instead of shadows; shadows only on floating layers.
- Attribution line on every section: agent, `tool`, sealed hash.
- Every state (hover, focus-visible, disabled, pressed) is drawn; motion only marks state.

## Colors

Cool blue-grey neutrals with one clinical blue and three verdict hues, each paired with a pale wash of the same hue.

### Primary
- **Clinical Blue**: the only call-to-action fill (primary buttons, Approve, Download report), links, focus rings, running state. Hover deepens to **Clinical Blue Deep**; **Clinical Blue Wash** marks active step, pressed chip/segment and drag-over.

### Neutral
- **Cool Paper**: app ground and zoom canvas. **Surface** (white): document, cards, popovers, inputs. **Aside**: rail, review column, hover fill, disabled fill.
- **Ink** body text; **Ink Heading** headings, values, filenames; **Ink Dim** secondary text, captions, table headers (5.59:1 on white, at least 4.75:1 on every wash).
- **Border** card and column edges; **Border Subtle** table/audit row rules; **Border Strong** control outlines and table header rule.

### Verdict
- **Verdict Green** approved / pass / best model / sealed-OK; **Amber** needs-you, warning, flagged %AUCextrap; **Verdict Red** rejected / fail / chain broken. Each pairs with its wash for pills and highlighted rows.

### Named Rules
**The One Call-to-Action Rule.** `btn-green` and `btn-primary` share Clinical Blue; the verdict is carried by the copy, never by a second CTA colour.
**The Colour-Plus-Word Rule.** No verdict is colour alone: pills, step trails, QC rows and audit status always carry an icon and a word.

## Typography

**UI Font:** Public Sans (fallback -apple-system, Segoe UI, sans-serif). **Mono:** IBM Plex Mono (fallback ui-monospace).

**Character:** One friendly grotesque for everything readable; mono is reserved for machine identities so a hash never reads as prose.

### Hierarchy
- **Headline** (600, 22px, -0.01em): document title only (h1).
- **Title** (600, 15px): section headings, review header, QC verdict heading (h2).
- **Body** (400, 13px, 1.45; prose replies 1.55): document text, buttons (500), table cells, step labels. Sections cap at 800px.
- **Label** (600, 12px, sentence case): rail group labels, card titles' subtitles at 400, pills, attribution lines, captions.
- **Stat** (600, 18px): dataset-profile stat values, with a 12px dim term beneath.
- **Mono** (400, 11-12px): ID chips, tool names, hashes, timings, subject IDs in the NCA table's first column.

### Named Rules
**The Sentence-Case Label Rule.** Labels are 12px sentence case at normal tracking; no uppercase tracked eyebrows anywhere in the shell.

## Layout

Fixed viewport frame (`html, body` never scroll; columns scroll internally). Grid: rail 248px / document `minmax(0,1fr)` / review 336px; topbar 52px across. Document scroller pads 28px 40px, sections gap 28px with a 1px rule and 28px bottom padding between them; content capped at 800px and centred, composer row aligned to the same 800px. Rail pads 20px 16px with 24px between groups; review pads 20px with 20px gaps. Spacing steps observed: 4, 8, 12, 16, 20, 28, 40 (6, 10, 14 inside compact controls).
- **≤1180px:** review column becomes a 336px fixed drawer under the topbar, toggled by a topbar button that shows an amber dot and "Review: decision needed" when a gate is open.
- **≤960px:** rail collapses to a 56px icon rail; step names, timings and states remain in the accessibility tree.

## Elevation & Depth

Flat by default; depth is tonal (cool-paper ground, aside columns, white document and cards, 1px borders).

### Shadow Vocabulary
- **Popover** (`0 8px 24px rgba(22,49,74,.14)`): model picker, token field, Actions and Export menus.
- **Drawer** (`-8px 0 24px rgba(22,49,74,.12)`): review drawer at ≤1180px only.
- **Modal** (`0 16px 48px rgba(0,0,0,.3)` over `rgba(22,49,74,.55)` scrim): chart zoom.

**The Float-Only Shadow Rule.** A shadow means the layer floats above the frame; resting cards and sections never cast one.

## Shapes

Gentle corners: 4px for mono chips, 6px for controls (buttons, chips, inputs, segmented groups, step rows), 8px for composer, dataset card, λz panels and popovers, 10px for review cards and modals, fully rounded pills (12-15px) for status. Borders are 1px; the upload zone alone is 1.5px dashed. No accent stripes: emphasis is a full wash fill or a full border colour change.

## Components

### Buttons
- **Shape:** 6px, 32px tall (36px in decision and composer rows), 0 12px padding, 13px/500.
- **Primary:** Clinical Blue fill, white 600 text; hover Clinical Blue Deep.
- **Ghost / default:** white with Border Strong outline; hover Aside fill.
- **Reject:** white, red 600 text, red-wash hover. **Link button:** blue text, underline on hover.
- **Disabled (all):** Ink Dim on Aside, Border, `not-allowed`. **Focus:** 2px Clinical Blue outline, 2px offset.

### Chips and segmented controls
- 28px, 12px text, white with Border Strong; hover Aside; `aria-pressed` fills Clinical Blue Wash with 500 weight. Segments share one outline with internal dividers; focus ring is inset (-2px).

### Cards / Containers
- **Decision card** (review column): white, 10px, 14px padding; title, reason textarea, Approve (flex 1) + Reject.
- **Outcome cards:** report (green title, mono filename, full-width Download), running (blue title), rejected (red title).
- **QC evidence:** verdict heading at 15px with icon; checklist rows 12px, warn/fail rows take the matching wash, bold name.
- **Section:** h2 + optional controls + right-aligned 12px attribution `agent · tool · sealed abc123`.

### Inputs / Fields
- White, Border Strong, 6-8px radius, 12-13px text, Ink Dim placeholder; focus-visible turns border Clinical Blue plus the global ring; disabled is Ink Dim on Aside.

### Navigation
- **Topbar:** logo, divider, breadcrumb with mono session chip, status pill (green/red dot + words), Model popover, token popover, drawer toggle.
- **Rail steps:** 30px rows, 14px icon column; done = Ink, active = blue wash, gate = amber wash with a pulsing amber dot and "Needs you", pending = Ink Dim.

### Audit trail (signature)
- 26px rows: mono index, agent, tool, mono short hash; reason sub-row in 11px; header status green "verified" / red "broken" with icon.

### Tables
- NCA table 13px tabular, 12px dim headers over a Border Strong rule, Border Subtle row rules, right-aligned numbers, mono subject column; over-limit cells amber 600 with a word badge.

### Motion
- Single 150ms `cubic-bezier(0.16,1,0.3,1)` for colour/border transitions and the drawer slide; spinners (0.6-0.8s) for running work; 1.6s opacity pulse only on the open gate dot. `prefers-reduced-motion: reduce` removes all animation and transition; smooth scroll only under `no-preference`.

## Do's and Don'ts

### Do:
- **Do** put every new result in a `ResultSection` with agent, tool and seal attribution.
- **Do** keep gate decisions in the review column's decision card, Approve in Clinical Blue, Reject as red text on white.
- **Do** pair every verdict colour with a lucide icon and a word.
- **Do** use mono only for IDs, tool names, hashes, timings and subject IDs.
- **Do** give every new control hover, focus-visible (2px blue ring) and disabled states from the button/chip vocabulary.

### Don't:
- **Don't** add a second CTA colour; green buttons are Clinical Blue.
- **Don't** cast shadows on resting cards or sections.
- **Don't** use uppercase tracked labels, accent stripes, gradients or glass.
- **Don't** set body or secondary text below 12px or lighter than Ink Dim; 11px is for mono identities only.
- **Don't** use Unicode glyphs (✓ ✗ ⚠ ▾ !) as icons; use lucide-react with an aria label or adjacent word.
