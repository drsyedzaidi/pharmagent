import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildDecisionBody, gateNeedsDecision, subjectStatus } from '../src/shell/lambdaZReview.ts';

const flagged = [{ id: 1 }, { id: 'B2' }];
const review = {
  status: 'revision_requested', accepted_subjects: [1], revision_subjects: ['B2'], flagged_subjects: flagged,
};

test('subject status follows the backend lists, ids compared as strings', () => {
  assert.equal(subjectStatus(review, '1'), 'accepted');
  assert.equal(subjectStatus(review, 'B2'), 'revision');
  assert.equal(subjectStatus(review, 3), 'pending');
  assert.equal(subjectStatus(null, 1), 'pending');
});

test('the gate asks for a decision only while open on flagged subjects', () => {
  assert.equal(gateNeedsDecision(review, flagged), true);
  assert.equal(gateNeedsDecision({ ...review, status: 'accepted' }, flagged), false);
  assert.equal(gateNeedsDecision({ ...review, status: 'not_required' }, []), false);
  assert.equal(gateNeedsDecision(null, flagged), false);
});

test('selecting every flagged subject omits subject_ids; a subset names them; blank notes are dropped', () => {
  assert.deepEqual(buildDecisionBody('accept', [1, 'B2'], flagged, '  '), { decision: 'accept' });
  assert.deepEqual(buildDecisionBody('revise', ['B2'], flagged, ' sparse tail '),
    { decision: 'revise', subject_ids: ['B2'], notes: 'sparse tail' });
});
