// Run-provenance quarantine banner and explicit HTTP 423 handling. Pure functions (unit-tested with `node --test`).

export type ProvenanceStatus = 'verified' | 'legacy_unverified' | 'mismatch' | null | undefined;

export interface BannerInfo {
  level: 'error' | 'warning';
  title: string;
  detail: string;
}

/** The banner a session needs, if any. Verified sessions need none. */
export function provenanceBanner(status: ProvenanceStatus, assurance: string | null | undefined): BannerInfo | null {
  if (status === 'mismatch') {
    return {
      level: 'error',
      title: 'Session quarantined: its run provenance does not match its audit record',
      detail:
        'This session is read-only. Analyses, uploads, state changes, reports and exports are blocked. ' +
        'You can still inspect the audit trail and the raw session state.' +
        (assurance ? ` Audit integrity mode: ${assurance}.` : ''),
    };
  }
  if (status === 'legacy_unverified') {
    return {
      level: 'warning',
      title: 'Legacy session: run provenance was not recorded (legacy_unverified)',
      detail:
        'This session was created before run provenance was recorded, so the software and conventions catalog ' +
        'that produced its results are unknown. Reports and exports are labelled legacy_unverified.',
    };
  }
  return null;
}

interface ErrorEnvelope {
  error?: { code?: string; message?: string; request_id?: string };
}

function parseEnvelope(text: string): ErrorEnvelope | null {
  try {
    const v: unknown = JSON.parse(text);
    return v && typeof v === 'object' ? (v as ErrorEnvelope) : null;
  } catch {
    return null;
  }
}

/** A user-facing message for a failed API response. 423 is always explained as a quarantine. */
export function describeApiError(status: number, bodyText: string): string {
  const env = parseEnvelope(bodyText);
  if (status === 423) {
    const rid = env?.error?.request_id ? ` (request ${env.error.request_id})` : '';
    return (
      'Blocked: this session is quarantined because its run provenance does not match its audit record. ' +
      'It is read-only: audit trail and raw state remain viewable; analyses, uploads, reports and exports are ' +
      `refused.${rid}`
    );
  }
  return env?.error?.message || bodyText || `HTTP ${status}`;
}
