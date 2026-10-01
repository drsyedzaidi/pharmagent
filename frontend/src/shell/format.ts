import type { AuditEntry, AuditIntegrityStatus } from '../types';
import type { Decision, StepView } from './types';

/** Names of the integrity checks that failed, in the order the backend
 *  applies them. MAC and anchor only exist in enforced mode; in hash-only
 *  mode the backend reports both false because nothing was checked. */
export function integrityFailures(integrity: AuditIntegrityStatus): string[] {
  const enforced = integrity.mode === 'enforced';
  return [
    !integrity.chain_ok && 'hash chain',
    enforced && !integrity.mac_ok && 'MAC',
    enforced && !integrity.anchor_ok && 'anchor',
  ].filter((f): f is string => typeof f === 'string');
}

/** First `n` characters of a hash, or '' when absent. */
export function shortHash(h?: string, n = 8): string {
  return h ? h.slice(0, n) : '';
}

/** Local wall-clock HH:MM for an ISO timestamp, '' when unparsable. */
export function hhmm(iso: string): string {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return '';
  const d = new Date(t);
  const pad = (v: number) => String(v).padStart(2, '0');
  return `${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/** Elapsed seconds → '<0.1 s' | '0.2 s' | '12 s' | '1.4 min'. */
export function fmtDuration(s: number): string {
  if (!Number.isFinite(s) || s < 0) return '';
  if (s < 0.1) return '<0.1 s';
  if (s < 10) return `${s.toFixed(1)} s`;
  if (s < 60) return `${Math.round(s)} s`;
  return `${(s / 60).toFixed(1)} min`;
}

/** Last audit entry sealed under `tool`, or null (also null when `tool` is undefined). */
export function latestEntryFor(audit: AuditEntry[], tool: string | undefined): AuditEntry | null {
  if (!tool) return null;
  for (let i = audit.length - 1; i >= 0; i--) {
    if (audit[i].tool === tool) return audit[i];
  }
  return null;
}

/** Workflow step keys whose backend tool name differs from the key. Every other
 *  step key equals its tool name (backend/app/workflows.py), and tools seal
 *  audit entries under `tool.name` (backend/app/tools/base.py). */
export const STEP_AUDIT_TOOL: Record<string, string> = {
  spaghetti_plot: 'generate_spaghetti_plot',
  qc_review: 'run_qc',
};

/** Per-step elapsed time = delta between consecutive audit seal timestamps
 *  (the entry for the step minus the entry before it). Only included when the
 *  step's entry exists, is not the first entry, and both timestamps parse. */
export function stepTimings(audit: AuditEntry[], steps: readonly StepView[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (const step of steps) {
    const entry = latestEntryFor(audit, STEP_AUDIT_TOOL[step.key] ?? step.key);
    if (!entry || entry.index <= 0) continue;
    const prev = audit[entry.index - 1];
    if (!prev) continue;
    const t1 = Date.parse(entry.timestamp);
    const t0 = Date.parse(prev.timestamp);
    if (Number.isNaN(t1) || Number.isNaN(t0)) continue;
    const label = fmtDuration((t1 - t0) / 1000);
    if (label) out[step.key] = label;
  }
  return out;
}

/** The decision the backend sealed for the gate at `gateStep`, read from the
 *  audit trail: the latest human_review entry ("approved after step N" /
 *  "rejected after step N", N = gateStep, backend/app/core/orchestrator.py
 *  resume_workflow) sealed after the gated step's own entry — a decision on
 *  this run's gate, not an earlier run's at the same step. null while the
 *  gate is undecided or unreached. */
export function sealedDecision(
  audit: AuditEntry[], steps: readonly StepView[], gateStep: number,
): Exclude<Decision, 'pending'> {
  const gate = steps[gateStep];
  if (!gate) return null;
  const gateSeal = latestEntryFor(audit, STEP_AUDIT_TOOL[gate.key] ?? gate.key);
  if (!gateSeal) return null;
  const suffix = ` after step ${gateStep}`;
  for (let i = audit.length - 1; i >= 0; i--) {
    const e = audit[i];
    if (e.index <= gateSeal.index) break;
    if (e.tool !== 'human_review' || !e.action.endsWith(suffix)) continue;
    if (e.action.startsWith('approved')) return 'approved';
    if (e.action.startsWith('rejected')) return 'rejected';
  }
  return null;
}
