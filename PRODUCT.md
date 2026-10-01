# Product

## Register

product

## Users

Pharmacometricians, clinical pharmacologists and modelling consultants running PK/PD analyses that will be reviewed by a regulator or a sponsor. They sit at a laptop or desk monitor, mostly in daylight, mid-task: a dataset is in hand and a report is due. On any screen the primary task is one of: get a dataset loaded and profiled, read a result the agents produced, decide at a review gate, or export. Secondary users are project managers and reviewers who open a session to check what was approved and when.

## Product Purpose

PharmAgent runs validated PK/PD analyses through AI agents that plan but never compute: every number comes from deterministic tools, every tool call is sealed in a SHA-256 audit chain, and a person approves each gated step. Success is a modeller finishing an NCA or population PK analysis in one sitting, with an audit trail a reviewer can replay, and never being surprised by a number.

## Brand Personality

Precise, calm, transparent. A modern workspace (light, roomy, friendly typography, subtle state-conveying motion) rather than a clinical instrument or a terminal. The tool should disappear into the analysis; attention goes to the data and the decision, not to the assistant.

## Anti-references

- Generic AI chatbot: chat bubbles as the primary surface, avatars, sparkles, purple gradients, an assistant persona front and centre.
- Six rows of action chips under a chat box.
- Marketing-dashboard polish inside the app: hero metrics, identical card grids, decorative gradients.
- Legacy pharma software: stacked modals, grey panels, 10px text.

## Design Principles

1. Results are the document, chat is the side channel. Completed steps read as sections of an analysis, attributed to the agent and tool that produced them, not as messages.
2. The decision is always in view. A review gate sits in a persistent review panel with the QC evidence beside it, never as a banner covering content.
3. Show the seal. Hashes, timings and "unanchored" caveats are visible at the point of use, in plain language.
4. Density with air. Tables and numbers are dense and tabular; everything around them has room.
5. Motion conveys state only: a step completing, a job running, a gate resolving.

## Accessibility & Inclusion

WCAG 2.2 AA: 4.5:1 for body and secondary text (the current `--text-dim` #8298AC fails at 12-13px on white and must be darkened), keyboard-operable gates and menus, visible focus, `prefers-reduced-motion` honoured. QC verdicts and diagnostics never rely on red vs green alone: pair colour with an icon and a word.
