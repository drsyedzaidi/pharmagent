// Pure logic for the λz accept/revise gate (POST /api/sessions/{sid}/lambda-z-review). Run with `node --test`.
import type { LambdaZReview, LambdaZReviewDecision, LambdaZFlaggedSubject } from '../types';

export type LzSubjectStatus = 'accepted' | 'revision' | 'pending';

const key = (id: string | number) => String(id);

/** Per-subject standing in the gate, from the backend's accepted/revision lists. */
export function subjectStatus(review: LambdaZReview | null | undefined, id: string | number): LzSubjectStatus {
  if ((review?.revision_subjects ?? []).some(i => key(i) === key(id))) return 'revision';
  if ((review?.accepted_subjects ?? []).some(i => key(i) === key(id))) return 'accepted';
  return 'pending';
}

/** The gate needs the analyst only while it is open on flagged subjects. */
export function gateNeedsDecision(review: LambdaZReview | null | undefined,
                                  flagged: LambdaZFlaggedSubject[] | null | undefined): boolean {
  if (!review || !(flagged?.length)) return false;
  return review.status === 'awaiting_analyst_decision' || review.status === 'revision_requested';
}

/** Request body: no ids = all currently flagged subjects (the backend's default). `selected` ⊆ flagged ids. */
export function buildDecisionBody(decision: LambdaZReviewDecision, selected: Array<string | number>,
                                  flagged: LambdaZFlaggedSubject[], notes: string) {
  const all = flagged.length > 0 && selected.length === flagged.length;
  return {
    decision,
    ...(selected.length && !all ? { subject_ids: selected } : {}),
    ...(notes.trim() ? { notes: notes.trim() } : {}),
  };
}
