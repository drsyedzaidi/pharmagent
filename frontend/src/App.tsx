import { useEffect, useRef, useState, useCallback } from 'react';
import type { DragEvent, ChangeEvent } from 'react';
import {
  FlaskConical, Upload, CheckCircle, XCircle,
  AlertTriangle, ChevronRight,
} from 'lucide-react';
import { api, setToken, getToken } from './api';
import { FlexplotPanel } from './flexplot';
import type { Decision, GateSigner, Marker, Outcome, VerifyNote, WorkflowName } from './shell/types';
import { MARKERS, REVIEW_DRAWER_MQ } from './shell/types';
import { hhmm, integrityFailures, latestEntryFor, sealedDecision, stepTimings } from './shell/format';
import { agentLabel } from './shell/agents';
import { Topbar } from './shell/Topbar';
import { Rail } from './shell/Rail';
import { ReviewPanel } from './shell/ReviewPanel';
import { ResultSection } from './shell/ResultSection';
import { DocumentHeader } from './shell/DocumentHeader';
import { DatasetProfile } from './shell/DatasetProfile';
import { ExportMenu } from './shell/ExportMenu';
import { Composer } from './shell/Composer';
import { ActionsMenu } from './shell/ActionsMenu';
import { PkModelEvidence } from './shell/PkModelEvidence';
import {
  BeCard,
  BootstrapCard,
  CalculatorsPanel,
  ClinsimCard,
  ClinpharmCard,
  CompartmentalCard,
  DiagnosticsCard,
  DosePropCard,
  DoseSweepCard,
  EngineComparisonCard,
  ExposureForestCard,
  ForecastCard,
  ForestCard,
  IndividualExposuresCard,
  LlmSettings,
  NcaLzPlot,
  NcaTable,
  NlmeCard,
  PediatricCard,
  PkModelCard,
  PopPkCard,
  PriorCheckCard,
  ProfileCard,
  QcCard,
  ReviewCard,
  RolesEditor,
  ScmCard,
  SimChart,
  SimestCard,
  SirCard,
  SkillsPanel,
  SpaghettiChart,
  SpecialPopCard,
  StatsAdviceCard,
  VpcCard,
  describeLlm,
  flaggedExtrap,
  ncaUnits,
  sig3,
  vpcStrataOptions,
} from './cards';
import type {
  Session, PharmState, ChatMessage, AuditEntry, AuditIntegrityStatus,
  WorkflowStatus, ContentBlock, PkModelDef, SkillDef,
  SpaghettiData, NcaPlotData, WorkflowResponse, WorkflowExecutedStep,
  WorkflowStartResponse,
} from './types';

/** Section title + attribution for every transcript result-card marker.
 *  Typed against the MARKERS tuple, so tsc fails if a marker loses its entry.
 *  `tool` is set ONLY where the backend registers a Tool of that exact name
 *  and the workflow/endpoint seals an audit entry under it (verified in
 *  backend/app/tools/*.py + workflows.py); the section then shows that
 *  entry's hash. Markers without a confirmed tool get agent-only attribution
 *  — never a client-side hash. `anchor` gives the section a DOM id. */
const CARD_META: Record<Marker, { title: string; agent: string; tool?: string; anchor?: string }> = {
  __SPAGHETTI__:     { title: 'Concentration-time',                 agent: 'data_manager', tool: 'generate_spaghetti_plot' },
  __NCA_TABLE__:     { title: 'Non-compartmental analysis',         agent: 'nca',          tool: 'compute_nca' },
  __NCA_LZ__:        { title: 'Terminal slope (λz)',                agent: 'nca',          tool: 'compute_nca', anchor: 'lz' },
  __QC_CARD__:       { title: 'QC review',                          agent: 'qc',           tool: 'run_qc' },
  __BE__:            { title: 'Bioequivalence',                     agent: 'be' },
  __DP__:            { title: 'Dose proportionality',               agent: 'dose_prop' },
  __CLINPHARM__:     { title: 'Clinical pharmacology calculator',   agent: 'clinpharm' },
  __STATS__:         { title: 'Statistical advice',                 agent: 'statistician' },
  __COMPARTMENTAL__: { title: 'Compartmental fit',                  agent: 'compartmental' },
  __POPPK__:         { title: 'Population PK summary',              agent: 'poppk' },
  __PENDING_TOOL__:  { title: 'Proposal · needs your confirmation', agent: 'simulator' },
  __PKMODEL__:       { title: 'Structural model fit',               agent: 'modeler',      tool: 'fit_pk_model', anchor: 'pk-model' },
  __VPC__:           { title: 'VPC / goodness-of-fit',              agent: 'modeler',      tool: 'run_vpc' },
  __NLME__:          { title: 'Population (NLME) fit',              agent: 'modeler',      tool: 'run_nlme' },
  __PRIORCHECK__:    { title: 'Prior check (Bayesian borrowing)',   agent: 'modeler' },
  __SCM__:           { title: 'Covariate model (SCM)',              agent: 'modeler',      tool: 'run_scm' },
  __ENGINES__:       { title: 'Cross-engine comparison',            agent: 'modeler',      tool: 'run_engine_comparison' },
  __FORECAST__:      { title: 'MAP / TDM forecast',                 agent: 'modeler' },
  __DIAG__:          { title: 'Residual diagnostics',               agent: 'modeler',      tool: 'run_diagnostics' },
  __FOREST__:        { title: 'Covariate forest',                   agent: 'modeler',      tool: 'run_covariate_forest' },
  __SIMEST__:        { title: 'Trial-design precision check',       agent: 'simulator',    tool: 'run_simest' },
  __BOOTSTRAP__:     { title: 'Bootstrap uncertainty',              agent: 'simulator',    tool: 'run_bootstrap' },
  __SIR__:           { title: 'SIR uncertainty',                    agent: 'simulator',    tool: 'run_sir' },
  __PROFILE__:       { title: 'Likelihood profiling',               agent: 'simulator',    tool: 'run_profile' },
  __SWEEP__:         { title: 'Dose sweep',                         agent: 'simulator' },
  __CLINSIM__:       { title: 'Clinical trial simulation',          agent: 'simulator' },
  __EXPFOREST__:     { title: 'Exposure covariate forest',          agent: 'simulator' },
  __SPECIALPOP__:    { title: 'Special-population simulation',      agent: 'simulator' },
  __INDIVEXP__:      { title: 'Individual exposures',               agent: 'simulator' },
  __PEDIATRIC__:     { title: 'Pediatric dose-finding',             agent: 'simulator' },
  __SIM__:           { title: 'Forward simulation',                 agent: 'simulator' },
  __REVIEW__:        { title: 'Adversarial review',                 agent: 'reviewer',     tool: 'adversarial_review' },
  __REPORT__:        { title: 'Report',                             agent: 'report' },
};

const STEPS = [
  { key: 'load_dataset',       label: 'Load dataset' },
  { key: 'profile_pk_dataset', label: 'Profile PK data' },
  { key: 'validate_cdisc',     label: 'Validate format' },
  { key: 'spaghetti_plot',     label: 'Spaghetti plot' },
  { key: 'compute_nca',        label: 'Compute NCA' },
  { key: 'adversarial_review', label: 'Adversarial review' },
  { key: 'qc_review',          label: 'QC review',  gate: true },
  { key: 'generate_report',    label: 'Generate report' },
] as const;

const MODELING_STEPS = [
  { key: 'load_dataset',          label: 'Load dataset' },
  { key: 'profile_pk_dataset',    label: 'Profile PK data' },
  { key: 'fit_pk_model',          label: 'Fit structural models' },
  { key: 'run_engine_comparison', label: 'Cross-engine comparison' },
  { key: 'adversarial_review',    label: 'Adversarial review', gate: true },
] as const;

const POPPK_FULL_STEPS = [
  { key: 'load_dataset',         label: 'Load dataset' },
  { key: 'profile_pk_dataset',   label: 'Profile PK data' },
  { key: 'validate_cdisc',       label: 'Validate format' },
  { key: 'spaghetti_plot',       label: 'Spaghetti plot' },
  { key: 'fit_pk_model',         label: 'Compare structural models', gate: true },
  { key: 'run_nlme',             label: 'Population (NLME) fit' },
  { key: 'run_scm',              label: 'Covariate model (SCM)' },
  { key: 'run_diagnostics',      label: 'Residual diagnostics' },
  { key: 'run_covariate_forest', label: 'Covariate forest' },
  { key: 'run_vpc',              label: 'VPC / goodness-of-fit' },
  { key: 'adversarial_review',   label: 'Adversarial review', gate: true },
  { key: 'generate_report',      label: 'Generate report' },
] as const;

/** Sidebar presentation per workflow — keeps the step tracker in one place. */
const WORKFLOW_UI: Record<WorkflowName, { title: string; steps: readonly { key: string; label: string; gate?: boolean }[] }> = {
  nca_full:       { title: 'NCA Workflow',        steps: STEPS },
  poppk_modeling: { title: 'Modeling Workflow',   steps: MODELING_STEPS },
  poppk_full:     { title: 'Population PK Workflow', steps: POPPK_FULL_STEPS },
};

/** Steps that submit a real population fit — minutes of compute, so resuming
 *  into one is polled as a background job rather than awaited inline. */
const HEAVY_STEPS = new Set<string>([
  'run_nlme', 'run_scm', 'run_engine_comparison', 'run_simest',
]);

/** While a workflow leg runs as a job, re-read the audit trail every this
 *  many poll ticks (1.5 s each) after the first, so the seals the leg adds
 *  reach the rail and the review panel before the leg ends. */
const AUDIT_REFRESH_EVERY_TICKS = 20;

/** Result card each workflow step renders once it has run, keyed by tool, with
 *  the PharmState slot the card reads. Mirrors what the sidebar chips push, so
 *  a leg that ran as a background job (the NLME leg) shows the same cards as
 *  one that ran inline. Steps missing here (and steps whose card is pushed
 *  from state below) get no message: a step with a section is not narrated,
 *  and the rest go to the document's folded run log. */
const STEP_CARD: Record<string, { marker: Marker; key: keyof PharmState }> = {
  run_nlme:             { marker: '__NLME__',   key: 'nlme_results' },
  run_scm:              { marker: '__SCM__',    key: 'scm_results' },
  run_diagnostics:      { marker: '__DIAG__',   key: 'diagnostics_results' },
  run_covariate_forest: { marker: '__FOREST__', key: 'forest_results' },
  run_vpc:              { marker: '__VPC__',    key: 'vpc_results' },
};

/** Human-review banner subtitle: what the gated step just finished and what
 *  approval runs next. Phrased per step key, falling back to the step labels
 *  so a new gate never shows another gate's sentence. */
const GATE_DONE: Record<string, string> = {
  qc_review: 'QC complete',
  fit_pk_model: 'Structural comparison complete',
  adversarial_review: 'Adversarial review complete',
};
const GATE_NEXT: Record<string, string> = {
  generate_report: 'generate the DOCX report',
  run_nlme: 'run the population fit',
};
/** What approval runs, as a chain — named once here, not restated in the title. */
const GATE_CHAIN: Record<string, string> = {
  generate_report: 'DOCX report',
  run_nlme: 'NLME → SCM → diagnostics → forest → VPC',
};
/** Reason placeholder per gated step, so each gate shows an example of its own. */
const GATE_PLACEHOLDER: Record<string, string> = {
  qc_review: 'e.g. Subject 1 extrapolation accepted; sparse terminal sampling',
  fit_pk_model: 'e.g. 1-cmt oral accepted; lowest AIC, all converged',
  adversarial_review: 'e.g. Reviewer concerns addressed; fit accepted',
};

function gateSubtitle(workflow: WorkflowName, gateIndex: number): string {
  const steps = WORKFLOW_UI[workflow].steps;
  const gate = steps[gateIndex];
  if (!gate) return 'Approve to continue the workflow.';
  const done = GATE_DONE[gate.key] ?? `${gate.label} complete`;
  const next = steps[gateIndex + 1];
  return next ? `${done}. Next: ${GATE_CHAIN[next.key] ?? next.label}.` : `${done}.`;
}

/** A step summary that opens with its own label ("Adversarial review: GOAL
 *  MET…") is shown without it: the run log entry is already headed by the
 *  label, so repeating it would read as a doubled prefix. */
function stripLabelPrefix(label: string, summary: string): string {
  const prefix = `${label}:`;
  if (!summary.toLowerCase().startsWith(prefix.toLowerCase())) return summary;
  return summary.slice(prefix.length).trimStart();
}

// `snap` freezes the PharmState slice a result card reads, so re-running an
// analysis later does not retroactively rewrite earlier cards in the transcript.
// `seal` is the audit entry hash bound to that snapshot by `bindSeals` (once,
// from the audit fetched after the card's own response); `seq` is the push
// order pushMsg stamps so a fetch sent earlier can never bind a later card.
type DisplayMsg = {
  role: string; content: string; agent?: string; tool?: string; id: string;
  snap?: PharmState | null; seal?: string; seq?: number;
};
const newMsgId = () => `${Date.now()}-${Math.random()}`;

/** Bind every still-unsealed snapshot card pushed no later than `seqAtFetch`
 *  to the latest audit entry sealed under its tool in `audit` — the entry of
 *  the run that produced the card's snapshot, since that audit was fetched
 *  after the card's response landed. A card is bound once and never rebound,
 *  so a later re-run (new card, new snapshot) leaves the older card's seal on
 *  the older numbers. Returns `msgs` itself when nothing binds. */
function bindSeals(msgs: DisplayMsg[], audit: AuditEntry[], seqAtFetch: number): DisplayMsg[] {
  let changed = false;
  const out = msgs.map(m => {
    if (!m.snap || m.seal !== undefined || (m.seq ?? Infinity) > seqAtFetch) return m;
    const marker = (MARKERS as readonly string[]).includes(m.content) ? m.content as Marker : null;
    const hash = marker ? latestEntryFor(audit, CARD_META[marker].tool)?.entry_hash : undefined;
    if (!hash) return m;
    changed = true;
    return { ...m, seal: hash };
  });
  return changed ? out : msgs;
}

function MessageBubble({ msg, agent }: { msg: DisplayMsg; agent?: string }) {
  if (msg.role === 'user') {
    return <div className="msg user">You · {msg.content}</div>;
  }
  return (
    <div className="msg agent">
      {agent && <span className="msg-agent">{agentLabel(agent)}</span>}
      <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
      {msg.tool && (
        <div className="tool-chip">
          <ChevronRight size={10} /> {msg.tool}
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [state, setState] = useState<PharmState | null>(null);
  const [wfStatus, setWfStatus] = useState<WorkflowStatus>('idle');
  const [activeWorkflow, setActiveWorkflow] = useState<WorkflowName>('nca_full');
  const [messages, setMessages] = useState<DisplayMsg[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [drag, setDrag] = useState(false);
  const [audit, setAudit] = useState<AuditEntry[]>([]);
  const [auditIntegrity, setAuditIntegrity] = useState<AuditIntegrityStatus | null>(null);
  const [healthy, setHealthy] = useState<boolean | null>(null);
  /** /api/health `auth`: null until known. Drives the gate's attribution copy. */
  const [authMode, setAuthMode] = useState<'required' | 'open' | null>(null);
  const [llmLabel, setLlmLabel] = useState<string>('');
  const [llmOpen, setLlmOpen] = useState(false);
  /** Client-side gate wait only: 'pending' from the moment a gate arrives
   *  until the next run (or a rejection) clears it. Approved / Rejected are
   *  never held here — `decisionView` below reads them from the sealed
   *  human_review audit entry, so nothing can show a decision the backend
   *  did not record. */
  const [decision, setDecision] = useState<Decision>(null);
  /** A gate decision is in flight (its POST unanswered): the gate stays up
   *  with its reason, and the review pill shows progress instead. */
  const [deciding, setDeciding] = useState(false);
  const [verifyNote, setVerifyNote] = useState<VerifyNote | null>(null);
  /** ≤1180px the review column is a drawer; this is its open state. */
  const [reviewOpen, setReviewOpen] = useState(false);
  const [calcOpen, setCalcOpen] = useState(false);
  const [currentStep, setCurrentStep] = useState(-1);
  /** Index of the step the workflow is paused at (from the gate payload). */
  const [gateStep, setGateStep] = useState(-1);
  /** Executed workflow steps that put no section in the document (load,
   *  validate, adversarial review…): shown once, folded, as the run log. */
  const [runLog, setRunLog] = useState<WorkflowExecutedStep[]>([]);
  /** Id of the upload-time "Dataset loaded" note; dropped once the workflow's
   *  own load_dataset step has reported the same fact. */
  const loadedNoteId = useRef('');
  const [pkModels, setPkModels] = useState<PkModelDef[]>([]);
  const [selectedModel, setSelectedModel] = useState('oral_1cmt');
  const [simDose, setSimDose] = useState(100);
  const [simTau, setSimTau] = useState(24);
  const [simNDoses, setSimNDoses] = useState(1);
  const [simTmax, setSimTmax] = useState<number | ''>('');
  const [sweepDoses, setSweepDoses] = useState('');
  const [errorModel, setErrorModel] = useState('proportional');
  const [jobNote, setJobNote] = useState('');
  const [fcDose, setFcDose] = useState(100);
  const [fcTau, setFcTau] = useState(24);
  const [fcLevels, setFcLevels] = useState('');
  const [fcTarget, setFcTarget] = useState('');
  const [fcMetric, setFcMetric] = useState('cmin');
  const [seN, setSeN] = useState(20);
  const [seObsT, setSeObsT] = useState('0.5,1,2,4,8,12,24');
  const [seDose, setSeDose] = useState(100);
  const [seNRep, setSeNRep] = useState(5);
  const [seShowConfirm, setSeShowConfirm] = useState(false);
  const [token, setTokenState] = useState(getToken());
  const [showRoles, setShowRoles] = useState(false);
  const [skills, setSkills] = useState<SkillDef[]>([]);
  const [showSkills, setShowSkills] = useState(false);
  const [showFlexplot, setShowFlexplot] = useState(false);
  /** Actions popover above the composer (quick-action rows live inside it). */
  const [actionsOpen, setActionsOpen] = useState(false);
  const actionsRef = useRef<HTMLButtonElement>(null);
  const messagesEnd = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const shownMarkers = useRef(new Set<string>());

  const scrollBottom = () => messagesEnd.current?.scrollIntoView({ behavior: 'smooth' });
  useEffect(() => { scrollBottom(); }, [messages]);

  useEffect(() => {
    api.health()
      .then(h => { setHealthy(h.status === 'ok'); setLlmLabel(h.llm ?? ''); setAuthMode(h.auth ?? null); })
      .catch(() => setHealthy(false));
    api.createSession()
      .then(s => setSession(s))
      .catch(console.error);
    api.listPkModels()
      .then(r => setPkModels(r.models))
      .catch(() => { /* library unavailable */ });
  }, []);

  // Accept an optional `id` (most call sites pass `id: ''` as a placeholder) and
  // assign a fresh unique id when none is given, so React keys stay stable and
  // distinct; a caller passes its own (newMsgId) only to remove the row later.
  /** Push order of the last message (see DisplayMsg.seq); read synchronously
   *  at pushMsg time, before React commits the push. */
  const msgSeq = useRef(0);
  const pushMsg = useCallback((m: Omit<DisplayMsg, 'id'> & { id?: string }) => {
    const seq = ++msgSeq.current;
    setMessages(prev => [...prev, { ...m, id: m.id || newMsgId(), seq }]);
  }, []);

  function extractMessages(raw: ChatMessage[], agent: string): DisplayMsg[] {
    const out: DisplayMsg[] = [];
    for (const m of raw) {
      // Backend agents emit bare strings (AgentResult.messages: list[str]);
      // render each as an assistant reply attributed to the routing agent.
      if (typeof m === 'string') {
        if (m.trim()) out.push({ role: 'assistant', content: m, agent, id: '' });
      } else if (typeof m.content === 'string') {
        if (m.content.trim()) out.push({ role: m.role, content: m.content, agent, id: '' });
      } else if (Array.isArray(m.content)) {
        for (const block of m.content as ContentBlock[]) {
          if (block.type === 'text' && block.text?.trim()) {
            out.push({ role: m.role, content: block.text, agent, id: '' });
          } else if (block.type === 'tool_use') {
            out.push({ role: 'assistant', content: `Running: ${block.name}`, agent, tool: block.name, id: '' });
          }
        }
      }
    }
    return out.map(m => ({ ...m, id: `${Date.now()}-${Math.random()}` }));
  }

  async function refreshAudit() {
    if (!session) return;
    try {
      const a = await api.getAudit(session.id);
      setAudit(a.entries);
      setAuditIntegrity(a.integrity);
    } catch { /* best-effort */ }
  }

  // Audit freshness: seals and step timings follow every state change (chat,
  // tool and calculator responses), not only workflow gates. Async then-branch
  // only; the `alive` flag drops a stale response after unmount/re-run. This
  // fetch is sent after the response that changed `state` landed, so it is
  // the one that binds the cards pushed with that response (bindSeals);
  // `seqAtFetch` keeps it from binding a card pushed after it was sent.
  useEffect(() => {
    if (!session || !state) return;
    let alive = true;
    const seqAtFetch = msgSeq.current;
    api.getAudit(session.id).then(r => {
      if (!alive) return;
      setAudit(r.entries);
      setAuditIntegrity(r.integrity);
      setMessages(prev => bindSeals(prev, r.entries, seqAtFetch));
      setVerifyNote(null);
    }).catch(() => { /* best-effort */ });
    return () => { alive = false; };
  }, [session, state]);

  /** Explicit chain check from the audit trail: re-reads the chain and reports
   *  the backend's own integrity verdict beside the link. */
  async function verifyChain() {
    if (!session) return;
    try {
      const r = await api.getAudit(session.id);
      setAudit(r.entries);
      setAuditIntegrity(r.integrity);
      const { mode, chain_ok, verified } = r.integrity;
      if (verified) {
        setVerifyNote({ tone: 'ok', text: 'Chain verified' });
      } else if (mode === 'hash_only' && chain_ok) {
        // Hash-only (dev) mode has no MAC or anchor: structure is all we can say.
        setVerifyNote({ tone: 'dim', text: 'Chain structure intact (unauthenticated)' });
      } else {
        const failed = integrityFailures(r.integrity);
        setVerifyNote({
          tone: 'bad',
          text: failed.length ? `Chain not verified: ${failed.join(', ')} failed` : 'Chain not verified',
        });
      }
    } catch {
      setVerifyNote({ tone: 'bad', text: 'Chain not verified' });
    }
  }

  /** v1: a fresh session is a page reload — App holds 40+ state slots plus the
   *  `shownMarkers` ref, so an in-place reset is a follow-up. */
  function newSession() {
    window.location.reload();
  }

  function handleFiles(f: File) {
    setFile(f);
  }
  function onFileChange(e: ChangeEvent<HTMLInputElement>) {
    if (e.target.files?.[0]) handleFiles(e.target.files[0]);
  }
  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDrag(false);
    if (e.dataTransfer.files[0]) handleFiles(e.dataTransfer.files[0]);
  }

  /** The server decides whether a workflow leg runs inline or as a job (any real
   *  population fit goes to the queue). Accept either shape: poll a job handle,
   *  then render the WorkflowResponse. Fixes "Cannot read properties of
   *  undefined (reading 'current_step')" on Run Modeling + Engines. */
  async function settleWorkflow(res: WorkflowStartResponse, note = 'Population fit running…') {
    if ('job_id' in res && res.job_id) {
      if (!session) return;
      handleWorkflowResponse(await pollWorkflowJob(session.id, res.job_id, note));
      return;
    }
    handleWorkflowResponse(res as WorkflowResponse);
  }

  /** Poll a workflow leg the backend queued. The audit trail is re-read on
   *  the first tick and periodically after it, so the seals the leg adds —
   *  the human_review entry first — reach the rail and the review pill while
   *  the leg runs: both read the sealed decision, never the click. */
  async function pollWorkflowJob(sid: string, jobId: string, note: string): Promise<WorkflowResponse> {
    let ticks = 0;
    try {
      return await api.pollJob<WorkflowResponse>(sid, jobId, s => {
        setJobNote(`${note} ${s}s (several real fits — this can take minutes)`);
        ticks += 1;
        if (ticks === 1 || ticks % AUDIT_REFRESH_EVERY_TICKS === 0) void refreshAudit();
      });
    } finally {
      setJobNote('');
    }
  }

  /** A leg the backend accepted lost its client-side tracking — the job poll
   *  failed after its retries, or the job itself reported an error. The
   *  backend session is the truth: reload its state and audit so the document
   *  and the trail show what was actually sealed, then settle the status from
   *  that state — complete when the workflow's last step ran, otherwise idle
   *  so the run can be repeated. Never a dead 'error'. */
  async function reconcileRun(err: Error, workflow: WorkflowName) {
    pushMsg({
      role: 'assistant', agent: 'supervisor', id: '',
      content: `Error: ${err.message} — session state reloaded; the audit trail shows what was sealed.`,
    });
    if (!session) return;
    let finished = false;
    try {
      const st = await api.getState(session.id);
      setState(st);
      setCurrentStep(st.current_step ?? -1);
      finished = st.workflow_name === workflow
        && (st.current_step ?? -1) >= WORKFLOW_UI[workflow].steps.length;
    } catch { /* the audit refresh below still reconciles the trail */ }
    await refreshAudit();
    setWfStatus(finished ? 'complete' : 'idle');
  }

  function handleWorkflowResponse(res: WorkflowResponse) {
    setState(res.state);
    setCurrentStep(res.state.current_step ?? -1);

    if (res.messages) {
      const agent = res.state.last_agent ?? 'supervisor';
      extractMessages(res.messages, agent).forEach(m => pushMsg(m));
    }

    // Backend tools whose result section this leg puts in the document. A step
    // that has one is not narrated as well — the section's attribution line
    // already names agent, tool and seal; the rest go to the folded run log.
    const sectioned = new Set<string>();
    if (res.state.dataset_metadata) sectioned.add('profile_pk_dataset');
    const pushCard = (marker: Marker, agent: string) => {
      pushMsg({ role: 'assistant', content: marker, agent, id: '', snap: res.state });
      const tool = CARD_META[marker].tool;
      if (tool) sectioned.add(tool);
    };

    // One result card per step the leg ran. `executed` is the only per-step
    // record a workflow leg returns — a leg polled from a job (NLME → SCM →
    // diagnostics → forest → VPC) arrives here exactly like an inline one, so
    // this is what keeps the document complete. Every card below is pushed
    // once per run: `shownMarkers` is cleared by uploadAndRun and each template
    // tool runs once, so a resumed leg (gate approve/reject) must not re-emit a
    // section the page already shows. The key is the marker, not the audit
    // seal — `audit` is refreshed after these pushes, so a seal is not known here.
    const executed = res.executed ?? [];
    for (const step of executed) {
      const agent = step.agent || res.state.last_agent || 'supervisor';
      const card = STEP_CARD[step.tool];
      if (card && res.state[card.key] && !shownMarkers.current.has(card.marker)) {
        shownMarkers.current.add(card.marker);
        pushCard(card.marker, agent);
      }
    }

    if (res.state.spaghetti_data && !shownMarkers.current.has('spaghetti')) {
      shownMarkers.current.add('spaghetti');
      pushCard('__SPAGHETTI__', 'data_manager');
    }
    if (res.state.nca_summary && !shownMarkers.current.has('nca')) {
      shownMarkers.current.add('nca');
      pushCard('__NCA_TABLE__', 'nca');
    }
    if (res.state.nca_plot_data && !shownMarkers.current.has('nca_lz')) {
      shownMarkers.current.add('nca_lz');
      pushCard('__NCA_LZ__', 'nca');
    }
    if (res.state.pk_model_results?.status === 'ok' && !shownMarkers.current.has('pkmodel')) {
      shownMarkers.current.add('pkmodel');
      pushCard('__PKMODEL__', 'modeler');
    }
    if (res.state.engine_comparison_results && !shownMarkers.current.has('engines')) {
      shownMarkers.current.add('engines');
      pushCard('__ENGINES__', 'modeler');
    }
    if (res.state.qc_verdict && !shownMarkers.current.has('qc')) {
      shownMarkers.current.add('qc');
      pushCard('__QC_CARD__', 'qc');
    }

    // Steps with no section of their own are logged, not narrated; the upload
    // note is dropped once load_dataset has reported the same fact.
    const unsectioned = executed.filter(s => !sectioned.has(s.tool));
    if (unsectioned.length) setRunLog(prev => [...prev, ...unsectioned]);
    if (loadedNoteId.current && executed.some(s => s.tool === 'load_dataset')) {
      const noteId = loadedNoteId.current;
      loadedNoteId.current = '';
      setMessages(prev => prev.filter(m => m.id !== noteId));
    }

    if (res.status === 'awaiting_review') {
      setGateStep(res.review?.after_step ?? (res.state.current_step ?? 0) - 1);
      setDecision('pending');
      setWfStatus('awaiting_review');
      // ≤1180px the panel is a closed drawer: a gate opens it, so a paused run
      // is never signalled by the topbar toggle alone (ReviewPanel then focuses
      // its heading in the same commit).
      if (window.matchMedia(REVIEW_DRAWER_MQ).matches) setReviewOpen(true);
      refreshAudit();
    } else if (res.status === 'complete') {
      setWfStatus('complete');
      // Same drawer rule: the report / complete card lives only in the panel
      // (chat suppresses __REPORT__), so a narrow viewport must open it.
      if (window.matchMedia(REVIEW_DRAWER_MQ).matches) setReviewOpen(true);
      if (res.state.report_path) {
        pushMsg({ role: 'assistant', content: '__REPORT__', agent: 'report', id: '' });
      }
      refreshAudit();
    } else if (res.status === 'rejected') {
      // Still maps to idle (re-run possible). The rejection is not held
      // client-side: the refreshed audit's sealed human_review entry is what
      // the rail and the review panel read (`decisionView`).
      setWfStatus('idle');
      setDecision(null);
      refreshAudit();
    } else {
      setWfStatus('idle');
    }
  }

  async function uploadAndRun(workflow: WorkflowName = 'nca_full') {
    if (!session || !file) return;
    shownMarkers.current.clear();
    setRunLog([]);
    loadedNoteId.current = '';
    setDecision(null);
    // A new run has no gate yet: an earlier run's sealed decision at the same
    // step must not read as this run's.
    setGateStep(-1);
    setActiveWorkflow(workflow);
    setLoading(true);
    setWfStatus('running');
    setCurrentStep(0);
    // No chat echo of the start: the document header already names the run.
    let accepted = false;
    try {
      const up = await api.uploadDataset(session.id, file);
      const meta = up.metadata;
      loadedNoteId.current = newMsgId();
      pushMsg({
        role: 'assistant',
        content: `Dataset loaded: ${meta['n_records']} records, ${meta['n_subjects']} subjects, ${meta['n_columns']} columns.`,
        agent: 'data_manager',
        id: loadedNoteId.current,
      });
      const res = await api.startWorkflow(session.id, meta['dataset_path'] as string ?? '', workflow);
      accepted = true;
      await settleWorkflow(res);
    } catch (e) {
      if (accepted) {
        await reconcileRun(e as Error, workflow);
      } else {
        // Nothing ran (a bad CSV, a refused start): back to idle, so the file
        // can be replaced and the run repeated without a reload.
        pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'supervisor', id: '' });
        setWfStatus('idle');
      }
    } finally {
      setJobNote('');
      setLoading(false);
    }
  }

  async function resume(approve: boolean, reason = '') {
    if (!session) return;
    setLoading(true);
    setDeciding(true);
    // The gate stays exactly as it is — wfStatus 'awaiting_review', decision
    // 'pending', the decision card and its reason mounted — until the backend
    // has taken the decision. A request that fails therefore leaves the gate
    // open to decide again, and nothing can read as Approved or Rejected
    // before the human_review entry exists (`decisionView` reads the seal).
    // Approving runs every remaining step in one call. If any of them is a real
    // population fit, poll a job instead of holding the request open — and say
    // what actually happens next rather than assuming the NCA shape.
    const remaining = WORKFLOW_UI[activeWorkflow].steps.slice(Math.max(currentStep, 0));
    const isLongLeg = approve && remaining.some(s => HEAVY_STEPS.has(s.key));
    const note = !approve ? 'Rejected — workflow stopped.'
      : isLongLeg ? 'Approved — running the population fit (NLME → SCM → diagnostics → forest → VPC).'
      : 'Approved — generating report.';
    pushMsg({ role: 'user', content: note, id: '' });
    let accepted = false;
    try {
      if (isLongLeg) {
        const { job_id } = await api.resumeWorkflowAsync(session.id, reason);
        accepted = true;
        setWfStatus('running');
        handleWorkflowResponse(await pollWorkflowJob(session.id, job_id, 'Population fit running…'));
        return;
      }
      const res = await api.resumeWorkflow(session.id, approve, reason);
      accepted = true;
      setWfStatus('running');
      await settleWorkflow(res);
    } catch (e) {
      if (accepted) {
        await reconcileRun(e as Error, activeWorkflow);
      } else {
        pushMsg({
          role: 'assistant', agent: 'supervisor', id: '',
          content: `Error: ${(e as Error).message} — the decision was not recorded; the gate is still open, decide again to retry.`,
        });
      }
    } finally {
      setDeciding(false);
      setJobNote('');
      setLoading(false);
    }
  }

  const AGENT_CARD: Record<string, string> = {
    be: '__BE__', dose_prop: '__DP__',
    compartmental: '__COMPARTMENTAL__', poppk: '__POPPK__', statistician: '__STATS__', clinpharm: '__CLINPHARM__',
    nca: '__NCA_TABLE__', qc: '__QC_CARD__',
  };

  async function sendChat(preset?: string) {
    const text = (preset ?? input).trim();
    if (!session || !text) return;
    if (!preset) setInput('');
    pushMsg({ role: 'user', content: text, id: '' });
    setLoading(true);
    try {
      const res = await api.chat(session.id, text);
      setState(res.state);
      extractMessages(res.messages ?? [], res.agent).forEach(m => pushMsg(m));
      // An expensive tool the agent PROPOSED (e.g. run_simest): nothing ran —
      // render the approve/reject card bound to this snapshot's proposal.
      if (res.pending_tool) {
        pushMsg({ role: 'assistant', content: '__PENDING_TOOL__', agent: res.agent, id: '', snap: res.state });
      }
      const marker = AGENT_CARD[res.agent];
      if (marker) pushMsg({ role: 'assistant', content: marker, agent: res.agent, id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'supervisor', id: '' });
    } finally {
      setLoading(false);
    }
  }

  function onKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
  }

  const PROPOSAL_CARD: Record<string, string> = {
    run_simest: '__SIMEST__', run_bootstrap: '__BOOTSTRAP__',
    run_sir: '__SIR__', run_profile: '__PROFILE__',
  };

  /** Human decision on a chat-proposed expensive tool. Approval is the tool's
   *  `confirm` and runs it as a background job (polled); rejection is inline. */
  async function decidePendingTool(approve: boolean) {
    if (!session) return;
    const tool = state?.pending_tool?.tool ?? 'proposed tool';
    setLoading(true);
    pushMsg({ role: 'user', content: `${approve ? 'Approve' : 'Reject'} ${tool}`, id: '' });
    try {
      const res = await api.decidePendingTool(session.id, approve);
      if (!approve || !res.job_id) {
        if (res.state) setState(res.state);
        pushMsg({ role: 'assistant', content: `${tool} rejected — nothing computed.`,
          agent: 'supervisor', id: '' });
        return;
      }
      // approved: the proposal is already cleared server-side; clear it locally
      // so the card stops offering buttons, then poll the job.
      setState(s => (s ? { ...s, pending_tool: null } : s));
      const job = await api.pollJob(session.id, res.job_id,
        s => setJobNote(`${tool} running as a background job… ${s}s`));
      setJobNote('');
      setState(job.state);
      pushMsg({ role: 'assistant', content: job.summary, agent: 'simulator', id: '' });
      const card = PROPOSAL_CARD[tool];
      if (card) pushMsg({ role: 'assistant', content: card, agent: 'simulator', id: '', snap: job.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'supervisor', id: '' });
    } finally { setJobNote(''); setLoading(false); }
  }

  async function runPkModel(body: { model_key?: string; compare?: boolean }) {
    if (!session) return;
    setLoading(true);
    const label = body.compare ? 'Compare PK models'
      : `Fit ${pkModels.find(m => m.key === body.model_key)?.label ?? body.model_key}`;
    pushMsg({ role: 'user', content: label, id: '' });
    try {
      const res = await api.runPkModel(session.id, body);
      setState(res.state);
      pushMsg({ role: 'assistant', content: res.summary, agent: 'modeler', id: '' });
      pushMsg({ role: 'assistant', content: '__PKMODEL__', agent: 'modeler', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'modeler', id: '' });
    } finally {
      setLoading(false);
    }
  }

  async function runSimulate() {
    if (!session) return;
    setLoading(true);
    pushMsg({ role: 'user', content: `Simulate ${simNDoses}×${simDose} q${simTau}h`, id: '' });
    try {
      const res = await api.simulate(session.id, {
        dose: simDose, tau: simTau, n_doses: simNDoses,
        ...(simTmax ? { tmax: simTmax } : {}),
      });
      setState(res.state);
      pushMsg({ role: 'assistant', content: res.summary, agent: 'simulator', id: '' });
      pushMsg({ role: 'assistant', content: '__SIM__', agent: 'simulator', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'simulator', id: '' });
    } finally {
      setLoading(false);
    }
  }

  async function applyRoles(overrides: Record<string, string>) {
    if (!session) return;
    setLoading(true);
    try {
      const res = await api.setRoles(session.id, overrides);
      setState(res.state);
      setShowRoles(false);
      pushMsg({ role: 'assistant', content: 'Column roles updated.', agent: 'data_manager', id: '' });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'data_manager', id: '' });
    } finally { setLoading(false); }
  }

  async function runVpc(opts?: { stratify_by?: string | null; dose_normalize?: boolean; x_by?: string;
    exposure_check?: boolean; blq_check?: boolean }) {
    if (!session) return;
    setLoading(true);
    const label = opts?.exposure_check ? ' — exposure predictive check'
      : opts?.blq_check ? ' — BLQ-incidence VPC'
      : opts?.stratify_by ? ` — stratified by ${opts.stratify_by}`
      : opts?.dose_normalize ? ' — dose-normalized' : '';
    pushMsg({ role: 'user', content: `VPC / goodness-of-fit${label}`, id: '' });
    try {
      const res = await api.vpc(session.id, opts);
      setState(res.state);
      pushMsg({ role: 'assistant', content: res.summary, agent: 'modeler', id: '' });
      pushMsg({ role: 'assistant', content: '__VPC__', agent: 'modeler', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'modeler', id: '' });
    } finally { setLoading(false); }
  }

  async function runReview() {
    if (!session) return;
    setLoading(true);
    pushMsg({ role: 'user', content: 'Adversarial review', id: '' });
    try {
      const res = await api.review(session.id);
      setState(res.state);
      const c = res.counts;
      const verdict = res.goal_met ? 'goal met' : 'findings block the goal';
      pushMsg({
        role: 'assistant',
        content: `Adversarial review (${res.iterations} pass${res.iterations === 1 ? '' : 'es'}): `
          + `${verdict} — ${c.CRITICAL} critical, ${c.HIGH} high, ${c.MEDIUM} medium, ${c.LOW} low.`,
        agent: 'reviewer', id: '',
      });
      pushMsg({ role: 'assistant', content: '__REVIEW__', agent: 'reviewer', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'reviewer', id: '' });
    } finally { setLoading(false); }
  }

  async function captureSkill() {
    if (!session) return;
    const name = window.prompt('Name this skill (reusable on new datasets):');
    if (!name) return;
    setLoading(true);
    try {
      const res = await api.captureSkill(session.id, { name: name.trim() });
      const tools = res.skill.steps.map(s => s.tool).join(' → ');
      pushMsg({
        role: 'assistant', agent: 'reviewer', id: '',
        content: `Captured skill **${res.skill.name}** (v${res.skill.version}): ${tools}. `
          + 'Replay it on a new dataset from the Skills panel.',
      });
      await refreshSkills();
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'reviewer', id: '' });
    } finally { setLoading(false); }
  }

  async function refreshSkills() {
    try {
      const res = await api.listSkills();
      setSkills(res.skills);
    } catch { /* skills are optional UI; ignore listing errors */ }
  }

  async function runSkill(name: string) {
    if (!state?.dataset_path) {
      pushMsg({ role: 'assistant', content: 'Load a dataset first to replay a skill.', agent: 'reviewer', id: '' });
      return;
    }
    setLoading(true);
    pushMsg({ role: 'user', content: `Replay skill: ${name}`, id: '' });
    try {
      const res = await api.runSkill(name, state.dataset_path);
      const ok = res.executed.filter(s => s.status === 'ok').length;
      const nca = res.state.nca_parameters?.length ?? 0;
      const rev = res.state.review_results;
      const revNote = rev ? ` Review ${rev.goal_met ? 'goal met' : `${rev.counts.CRITICAL + rev.counts.HIGH} blocker(s)`}.` : '';
      pushMsg({
        role: 'assistant', agent: 'reviewer', id: '',
        content: `Replayed **${name}** on the current dataset → new session \`${res.session_id}\` `
          + `(${ok}/${res.executed.length} steps ok${nca ? `, NCA n=${nca}` : ''}).${revNote}`,
      });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'reviewer', id: '' });
    } finally { setLoading(false); }
  }

  async function deleteSkill(name: string) {
    if (!window.confirm(`Delete skill "${name}"?`)) return;
    try {
      await api.deleteSkill(name);
      await refreshSkills();
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'reviewer', id: '' });
    }
  }

  async function runNlme(method: string, opts?: { prior_from?: string; prior_var?: number }) {
    if (!session) return;
    const label: Record<string, string> = {
      focei: 'FOCE-I only', saem: 'SAEM',
      focei_saem: 'FOCE-I (SAEM-seeded)', auto: 'Auto (escalating)',
    };
    const priorNote = opts?.prior_from
      ? ` + informative prior (MAP${opts.prior_var != null ? `, var ${opts.prior_var}` : ''})` : '';
    setLoading(true);
    pushMsg({ role: 'user', content: `NLME fit — ${label[method] ?? method} (${errorModel} error)${priorNote}`, id: '' });
    try {
      const { job_id } = await api.nlme(session.id, {
        method, error_model: errorModel,
        ...(opts?.prior_from ? { prior_from: opts.prior_from } : {}),
        ...(opts?.prior_var != null ? { prior_var: opts.prior_var } : {}),
      });
      const res = await api.pollJob(session.id, job_id,
        s => setJobNote(`Population fit running… ${s}s`));
      setJobNote('');
      setState(res.state);
      pushMsg({ role: 'assistant', content: res.summary, agent: 'modeler', id: '' });
      pushMsg({ role: 'assistant', content: '__NLME__', agent: 'modeler', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'modeler', id: '' });
    } finally { setJobNote(''); setLoading(false); }
  }

  async function runPriorCheck() {
    if (!session) return;
    setLoading(true);
    pushMsg({ role: 'user', content: 'Prior check (predictive band + shrinkage)', id: '' });
    try {
      const res = await api.priorCheck(session.id, { n_draws: 500 });
      setState(res.state);
      pushMsg({ role: 'assistant', content: res.summary, agent: 'modeler', id: '' });
      pushMsg({ role: 'assistant', content: '__PRIORCHECK__', agent: 'modeler', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'modeler', id: '' });
    } finally { setLoading(false); }
  }

  async function runScm() {
    if (!session) return;
    setLoading(true);
    pushMsg({ role: 'user', content: `Covariate search (SCM, ${errorModel} error)`, id: '' });
    try {
      const { job_id } = await api.scm(session.id, { error_model: errorModel });
      const res = await api.pollJob(session.id, job_id,
        s => setJobNote(`Covariate search running… ${s}s (this can take a few minutes)`));
      setJobNote('');
      setState(res.state);
      pushMsg({ role: 'assistant', content: res.summary, agent: 'modeler', id: '' });
      pushMsg({ role: 'assistant', content: '__SCM__', agent: 'modeler', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'modeler', id: '' });
    } finally { setJobNote(''); setLoading(false); }
  }

  async function runSimest() {
    if (!session) return;
    const obs_t = seObsT.split(',').map(s => Number(s.trim())).filter(Number.isFinite);
    if (!obs_t.length || seN < 2) return;
    setSeShowConfirm(false);
    setLoading(true);
    pushMsg({ role: 'user', content: `Simulation-estimation design check — N=${seN}, ${seNRep} replicate(s)`, id: '' });
    try {
      const { job_id } = await api.simest(session.id, {
        confirm: true,
        design: { n_subjects: seN, obs_t, dose: seDose, n_doses: 1 },
        n_rep: seNRep,
      });
      const res = await api.pollJob(session.id, job_id,
        s => setJobNote(`Simulation-estimation running… ${s}s (several real fits — this can take minutes)`));
      setJobNote('');
      setState(res.state);
      pushMsg({ role: 'assistant', content: res.summary, agent: 'simulator', id: '' });
      pushMsg({ role: 'assistant', content: '__SIMEST__', agent: 'simulator', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'simulator', id: '' });
    } finally { setJobNote(''); setLoading(false); }
  }

  async function runEngineComparison() {
    if (!session) return;
    setLoading(true);
    pushMsg({ role: 'user', content: 'Cross-engine comparison (FOCE-I vs nlmixr2 vs Monolix)', id: '' });
    try {
      const { job_id } = await api.engineComparison(session.id,
        { engines: ['pharmagent_focei', 'nlmixr2', 'monolix'] });
      const res = await api.pollJob(session.id, job_id,
        s => setJobNote(`Cross-engine comparison running… ${s}s (fits + external engine can take a minute)`));
      setJobNote('');
      setState(res.state);
      pushMsg({ role: 'assistant', content: res.summary, agent: 'modeler', id: '' });
      pushMsg({ role: 'assistant', content: '__ENGINES__', agent: 'modeler', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'modeler', id: '' });
    } finally { setJobNote(''); setLoading(false); }
  }

  async function runForecast() {
    if (!session) return;
    const measured = fcLevels.split(/[;\n]+/).map(s => s.trim()).filter(Boolean)
      .map(pair => {
        const [t, c] = pair.split(/[,\s]+/).map(Number);
        return { time: t, conc: c };
      })
      .filter(m => Number.isFinite(m.time) && Number.isFinite(m.conc));
    setLoading(true);
    pushMsg({ role: 'user', content: `MAP forecast — ${fcDose} q${fcTau}h, ${measured.length} level(s)`, id: '' });
    try {
      const body: Parameters<typeof api.forecast>[1] = { dose: fcDose, tau: fcTau, measured };
      const tgt = Number(fcTarget);
      if (fcTarget.trim() !== '' && Number.isFinite(tgt)) { body.target = tgt; body.target_metric = fcMetric; }
      const res = await api.forecast(session.id, body);
      setState(res.state);
      pushMsg({ role: 'assistant', content: res.summary, agent: 'modeler', id: '' });
      pushMsg({ role: 'assistant', content: '__FORECAST__', agent: 'modeler', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'modeler', id: '' });
    } finally { setLoading(false); }
  }

  async function runDiagnostics() {
    if (!session) return;
    setLoading(true);
    pushMsg({ role: 'user', content: 'Residual diagnostics (IWRES / NPDE)', id: '' });
    try {
      const res = await api.diagnostics(session.id);
      setState(res.state);
      pushMsg({ role: 'assistant', content: res.summary, agent: 'modeler', id: '' });
      pushMsg({ role: 'assistant', content: '__DIAG__', agent: 'modeler', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'modeler', id: '' });
    } finally { setLoading(false); }
  }

  async function runCovariateForest() {
    if (!session) return;
    setLoading(true);
    pushMsg({ role: 'user', content: 'Covariate forest plot', id: '' });
    try {
      const res = await api.forest(session.id);
      setState(res.state);
      pushMsg({ role: 'assistant', content: res.summary, agent: 'modeler', id: '' });
      pushMsg({ role: 'assistant', content: '__FOREST__', agent: 'modeler', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'modeler', id: '' });
    } finally { setLoading(false); }
  }

  async function downloadFullReport() {
    if (!session) return;
    setLoading(true);
    try {
      const res = await api.generateReport(session.id);
      await api.downloadReportFile(session.id, res.result.report_path);
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'report', id: '' });
    } finally { setLoading(false); }
  }

  async function exportCsv(kind: string) {
    if (!session) return;
    try {
      await api.exportCsv(session.id, kind);
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'data_manager', id: '' });
    }
  }

  async function exportCdisc() {
    if (!session) return;
    try {
      await api.downloadCdisc(session.id);
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'data_manager', id: '' });
    }
  }

  /** Review-panel "Download DOCX": the report already exists, so fetch it
   *  through the authenticated blob path (a bare <a href> cannot send the
   *  bearer header and 401s when auth is required). */
  async function downloadExistingReport() {
    if (!session || !state?.report_path) return;
    try {
      await api.downloadReportFile(session.id, state.report_path);
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'report', id: '' });
    }
  }

  async function exportControl(kind: 'nonmem' | 'mrgsolve') {
    if (!session) return;
    try {
      await api.exportControl(session.id, kind);
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'modeler', id: '' });
    }
  }

  async function runDoseSweep() {
    if (!session) return;
    setLoading(true);
    const doses = sweepDoses.split(',').map(s => Number(s.trim())).filter(n => n > 0);
    pushMsg({ role: 'user', content: `Dose sweep: ${doses.join(', ')}`, id: '' });
    try {
      const res = await api.doseSweep(session.id, {
        doses: doses.length ? doses : undefined, tau: simTau, n_doses: simNDoses,
        ...(simTmax ? { tmax: simTmax } : {}),
      });
      setState(res.state);
      pushMsg({ role: 'assistant', content: res.summary, agent: 'simulator', id: '' });
      pushMsg({ role: 'assistant', content: '__SWEEP__', agent: 'simulator', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'simulator', id: '' });
    } finally { setLoading(false); }
  }

  async function runClinsim(opts?: { doses?: number[]; metric?: string; threshold?: number | null;
    direction?: string; target_fraction?: number; n_subjects?: number; param_uncertainty?: boolean }) {
    if (!session) return;
    setLoading(true);
    pushMsg({ role: 'user', content: 'Clinical trial simulation (target attainment)', id: '' });
    try {
      const res = await api.clinsim(session.id, {
        ...(opts?.doses?.length ? { doses: opts.doses } : { dose: simDose }),
        tau: simTau, n_doses: simNDoses,
        ...(opts?.metric ? { metric: opts.metric } : {}),
        ...(opts && 'threshold' in opts ? { threshold: opts.threshold } : {}),
        ...(opts?.direction ? { direction: opts.direction } : {}),
        ...(opts?.target_fraction != null ? { target_fraction: opts.target_fraction } : {}),
        ...(opts?.n_subjects ? { n_subjects: opts.n_subjects } : {}),
        ...(opts?.param_uncertainty ? { param_uncertainty: true } : {}),
      });
      setState(res.state);
      pushMsg({ role: 'assistant', content: res.summary, agent: 'simulator', id: '' });
      pushMsg({ role: 'assistant', content: '__CLINSIM__', agent: 'simulator', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'simulator', id: '' });
    } finally { setLoading(false); }
  }

  async function runExposureForest() {
    if (!session) return;
    setLoading(true);
    pushMsg({ role: 'user', content: 'Exposure covariate forest', id: '' });
    try {
      const res = await api.exposureForest(session.id, { dose: simDose, tau: simTau, n_doses: simNDoses });
      setState(res.state);
      pushMsg({ role: 'assistant', content: res.summary, agent: 'simulator', id: '' });
      pushMsg({ role: 'assistant', content: '__EXPFOREST__', agent: 'simulator', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'simulator', id: '' });
    } finally { setLoading(false); }
  }

  async function runSpecialPop(opts?: { source?: string; stratify_by?: string | null }) {
    if (!session) return;
    setLoading(true);
    pushMsg({ role: 'user', content: 'Special-population exposure simulation', id: '' });
    try {
      const res = await api.specialPopulation(session.id, {
        dose: simDose, tau: simTau, n_doses: simNDoses,
        ...(opts?.source ? { source: opts.source } : {}),
        ...(opts?.stratify_by ? { stratify_by: opts.stratify_by } : {}),
      });
      setState(res.state);
      pushMsg({ role: 'assistant', content: res.summary, agent: 'simulator', id: '' });
      pushMsg({ role: 'assistant', content: '__SPECIALPOP__', agent: 'simulator', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'simulator', id: '' });
    } finally { setLoading(false); }
  }

  async function runIndividualExposures() {
    if (!session) return;
    setLoading(true);
    pushMsg({ role: 'user', content: 'Individual steady-state exposures', id: '' });
    try {
      const res = await api.individualExposures(session.id, { dose: simDose, tau: simTau, n_doses: simNDoses });
      setState(res.state);
      pushMsg({ role: 'assistant', content: res.summary, agent: 'simulator', id: '' });
      pushMsg({ role: 'assistant', content: '__INDIVEXP__', agent: 'simulator', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'simulator', id: '' });
    } finally { setLoading(false); }
  }

  async function runPediatric(opts?: { source?: string; wt_exponent_cl?: number | null; wt_exponent_v?: number | null }) {
    if (!session) return;
    setLoading(true);
    pushMsg({ role: 'user', content: 'Pediatric dose-finding simulation', id: '' });
    try {
      const res = await api.pediatricSimulation(session.id, {
        tau: simTau, n_doses: simNDoses,
        ...(opts?.source ? { source: opts.source } : {}),
        ...(opts?.wt_exponent_cl != null ? { wt_exponent_cl: opts.wt_exponent_cl } : {}),
        ...(opts?.wt_exponent_v != null ? { wt_exponent_v: opts.wt_exponent_v } : {}),
      });
      setState(res.state);
      pushMsg({ role: 'assistant', content: res.summary, agent: 'simulator', id: '' });
      pushMsg({ role: 'assistant', content: '__PEDIATRIC__', agent: 'simulator', id: '', snap: res.state });
    } catch (e) {
      pushMsg({ role: 'assistant', content: `Error: ${(e as Error).message}`, agent: 'simulator', id: '' });
    } finally { setLoading(false); }
  }

  const QUICK_ACTIONS = [
    { label: 'Dose proportionality', msg: 'run dose proportionality power model' },
    { label: 'Compartmental fit', msg: 'fit a one- and two-compartment model' },
    { label: 'Population PK', msg: 'population pk typical values and iiv' },
    { label: 'Bioequivalence', msg: 'run a bioequivalence assessment test vs reference' },
    { label: 'Stats advice', msg: 'how should I analyze this data: parametric or non-parametric, which test?' },
  ];
  const hasData = !!state?.dataset_id;
  const EXPORT_LABEL: Record<string, string> = {
    nca: 'NCA', be: 'BE', dose_prop: 'Dose-prop', pk_model: 'Model',
    nlme: 'NLME', dose_sweep: 'Dose sweep', vpc: 'VPC',
  };
  const exportKinds = (state ? [
    state.nca_parameters?.length ? 'nca' : null,
    state.be_results?.status === 'ok' ? 'be' : null,
    state.dose_prop_results?.status === 'ok' ? 'dose_prop' : null,
    state.pk_model_results?.status === 'ok' ? 'pk_model' : null,
    state.nlme_results?.status === 'ok' ? 'nlme' : null,
    state.dose_sweep_results?.status === 'ok' ? 'dose_sweep' : null,
    state.vpc_results?.status === 'ok' ? 'vpc' : null,
  ] : []).filter(Boolean) as string[];

  const canRunWorkflow = !!session && !!file && wfStatus === 'idle' && !loading;

  const wfSteps = WORKFLOW_UI[activeWorkflow].steps;
  const timings = stepTimings(audit, wfSteps);
  const datasetMeta = (state?.dataset_metadata ?? null) as
    { n_records?: unknown; n_subjects?: unknown; n_columns?: unknown } | null;
  /** What the backend will seal as the gate actor (main.py current_owner):
   *  token:<sha256[:16]> only when auth is required AND a non-blank token is
   *  set; with auth open the token is ignored and the entry is anonymous. */
  const gateSigner: GateSigner = authMode === 'open' ? 'open'
    : authMode === 'required' && token.trim() !== '' ? 'token'
    : 'anonymous';
  const statusLabel = healthy === null ? 'connecting…'
    : healthy ? `Backend online · ${describeLlm(llmLabel)}` : 'Backend offline';

  // ── Review panel inputs (all read from live state + the sealed audit) ──
  const gateDef = wfSteps[gateStep];
  const nextDef = wfSteps[gateStep + 1];
  const gate = wfStatus === 'awaiting_review' && gateDef ? {
    title: `Approve to ${nextDef ? (GATE_NEXT[nextDef.key] ?? `run "${nextDef.label}"`) : 'finish the workflow'}`,
    subtitle: gateSubtitle(activeWorkflow, gateStep),
    stepLabel: gateDef.label,
    placeholder: GATE_PLACEHOLDER[gateDef.key] ?? 'e.g. Evidence reviewed; accepted',
  } : null;
  /** What the rail and the review panel show for the gate: Approved / Rejected
   *  only from the sealed human_review entry for this run's gate; otherwise
   *  the client-side wait ('pending' at an open gate, and while the taken
   *  decision's leg runs). */
  const decisionView: Decision = sealedDecision(audit, wfSteps, gateStep) ?? decision;
  const longLeg = wfSteps.slice(Math.max(currentStep, 0)).some(s => HEAVY_STEPS.has(s.key));
  /** QC evidence belongs to the workflow that runs QC (nca_full's qc_review
   *  gate) — and to a chat "qc" turn, whose default workflow is that one. The
   *  backend never clears qc_verdict, so under a population workflow it is an
   *  earlier run's and must not shadow that run's gate evidence (fit_pk_model /
   *  adversarial_review below). */
  const runsQc = wfSteps.some(s => s.key === 'qc_review');
  const qc = runsQc && state?.qc_verdict ? {
    verdict: state.qc_verdict,
    checklist: state.qc_checklist,
    issues: state.qc_issues,
    hash: latestEntryFor(audit, 'run_qc')?.entry_hash ?? null,
    subjects: state.nca_parameters,
  } : null;
  /** Jump to the latest structural-model section in the document (id lands in
   *  its ResultSection; a re-run adds a newer section, so take the last). */
  const onSeeComparison = () => {
    const els = document.querySelectorAll<HTMLElement>('#pk-model');
    els[els.length - 1]?.scrollIntoView({ block: 'start' });
  };
  const evidence = gateDef?.key === 'adversarial_review' && state?.review_results
    ? <ReviewCard r={state.review_results} />
    : gateDef?.key === 'fit_pk_model' && state?.pk_model_results
      ? <PkModelEvidence r={state.pk_model_results}
          seal={latestEntryFor(audit, 'fit_pk_model')?.entry_hash ?? null}
          onSeeComparison={onSeeComparison} />
      : null;
  const outcome: Outcome | null =
    wfStatus === 'complete' && state?.report_path && session ? {
      kind: 'report',
      filename: state.report_path.split('/').pop() ?? 'report.docx',
      onDocx: () => { downloadExistingReport(); },
      onCsv: state.nca_parameters?.length ? () => { exportCsv('nca'); } : undefined,
      onCdisc: state.nca_parameters?.length ? () => { exportCdisc(); } : undefined,
    }
    : wfStatus === 'complete' ? { kind: 'complete' }
    : decisionView && wfStatus === 'running' ? {
      kind: 'running',
      note: decisionView === 'approved'
        ? (longLeg
          ? 'Approved — running the population fit (NLME → SCM → diagnostics → forest → VPC)'
          : 'Approved — generating report')
        : '',
    }
    : decisionView === 'rejected' && wfStatus === 'idle' ? {
      kind: 'rejected',
      stepLabel: gateDef?.label ?? 'the gate',
      entryIndex: [...audit].reverse().find(e => e.tool === 'human_review' && e.action.startsWith('rejected'))?.index ?? null,
      hint: activeWorkflow === 'nca_full'
        ? 'Refit λz or replace the dataset, then run again.'
        : 'Adjust the analysis, then run again.',
    }
    : null;
  /** Jump to the latest λz section in the document (id lands in S4; a re-run
   *  adds a newer section, so take the last) and focus its first control. */
  const onReviewLz = () => {
    const els = document.querySelectorAll<HTMLElement>('#lz');
    const el = els[els.length - 1];
    el?.scrollIntoView({ block: 'start' });
    (el?.querySelector('button') as HTMLElement | null)?.focus();
  };
  /** Hand focus to the review panel's heading. ≤1180px the panel is a drawer
   *  (visibility:hidden when closed), so open it first; ReviewPanel then
   *  focuses the heading itself when `open` flips. */
  const onReviewInPanel = () => {
    if (window.matchMedia(REVIEW_DRAWER_MQ).matches) setReviewOpen(true);
    document.getElementById('review-heading')?.focus();
  };
  /** Closing the drawer hands focus back to the topbar toggle (only rendered ≤1180px). */
  const closeReview = () => {
    setReviewOpen(false);
    document.getElementById('review-toggle')?.focus();
  };

  // ── Document (centre column) inputs ──
  /** Live seal for a marker's tool: the latest audit entry sealed under that
   *  backend tool name (refreshed by the state-keyed audit effect). Never a
   *  client-side hash; null when the marker has no registered tool. Only for
   *  cards that render live state — a snapshot card shows the seal bound to
   *  its own snapshot (bindSeals), so a re-run never relabels older numbers. */
  const sealFor = (k: Marker) => latestEntryFor(audit, CARD_META[k].tool)?.entry_hash ?? null;
  /** ResultSection attribution props for one marker message. */
  const sectionProps = (k: Marker, m: DisplayMsg) => {
    const meta = CARD_META[k];
    const seal = m.snap ? (m.seal ?? null) : sealFor(k);
    return {
      id: meta.anchor, title: meta.title, agent: m.agent ?? meta.agent,
      tool: meta.tool, seal, sealing: !!meta.tool && !seal,
    };
  };
  const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
  const nSubjects = num(datasetMeta?.n_subjects);
  const docTitle = (messages.length === 0 && !file) ? 'New analysis'
    : ({ nca_full: 'NCA analysis', poppk_modeling: 'Modeling analysis', poppk_full: 'Population PK analysis' })[activeWorkflow];
  const startedAt = session?.created_at ? hhmm(session.created_at) : '';
  const docSubtitle = [
    file?.name,
    nSubjects != null && `${nSubjects} subjects`,
    startedAt && `started ${startedAt}`,
    audit.length > 0 && `${audit.length} sealed entries`,
  ].filter(Boolean).join(' · ') || 'Upload a PK dataset and choose a workflow.';
  /** Dataset profile stats — only values that exist in PharmState (no dashes; units only when the dataset labels them). */
  const profileItems: { label: string; value: string }[] = [];
  if (state?.dataset_metadata) {
    const dq = (state.data_quality ?? {}) as Record<string, unknown>;
    const nRecords = num(datasetMeta?.n_records);
    const nObs = num(dq.n_observations);
    const blq = num(state.spaghetti_data?.blq_excluded) ?? num(state.nca_summary?.blq?.n_below_loq);
    const doses = (state.nca_parameters ?? []).map(r => r.dose).filter((d): d is number => num(d) != null);
    if (nSubjects != null) profileItems.push({ label: 'Subjects', value: String(nSubjects) });
    if (nRecords != null) profileItems.push({ label: 'Records', value: String(nRecords) });
    if (nObs != null) profileItems.push({ label: 'Observations', value: String(nObs) });
    if (blq != null) profileItems.push({ label: 'Below LLOQ, excluded', value: String(blq) });
    if (doses.length) {
      const lo = sig3(Math.min(...doses)), hi = sig3(Math.max(...doses));
      const doseUnit = ncaUnits(state).dose;
      profileItems.push({ label: 'Dose range', value: `${lo === hi ? lo : `${lo}–${hi}`}${doseUnit ? ` ${doseUnit}` : ''}` });
    }
  }
  const profileSeal = latestEntryFor(audit, 'profile_pk_dataset')?.entry_hash ?? null;
  /** Export menu items — the same handlers and conditions as the former export chips. */
  const exportItems = [
    { id: 'docx', label: 'Full report (DOCX)', onClick: () => { downloadFullReport(); } },
    ...exportKinds.map(k => ({ id: `csv-${k}`, label: `${EXPORT_LABEL[k] ?? k} CSV`, onClick: () => { exportCsv(k); } })),
    ...(state?.nca_parameters?.length
      ? [{ id: 'cdisc', label: 'CDISC ADaM (zip)', onClick: () => { exportCdisc(); } }] : []),
    ...(state?.nlme_results?.status === 'ok' ? [
      { id: 'nonmem', label: 'NONMEM (.ctl)', title: 'NONMEM control stream (.ctl) seeded from the population fit',
        onClick: () => { exportControl('nonmem'); } },
      { id: 'mrgsolve', label: 'mrgsolve (.cpp)', title: 'mrgsolve model (.cpp) seeded from the population fit',
        onClick: () => { exportControl('mrgsolve'); } },
    ] : []),
  ];

  return (
    <>
      <Topbar
        sessionId={session?.id ?? null}
        title={file?.name ?? 'New session'}
        healthy={healthy}
        statusLabel={statusLabel}
        llmOpen={llmOpen}
        onLlmToggle={() => setLlmOpen(o => !o)}
        onLlmClose={() => setLlmOpen(false)}
        llmPanel={<LlmSettings onApplied={l => setLlmLabel(l)} onClose={() => setLlmOpen(false)} />}
        token={token}
        onTokenChange={v => { setTokenState(v); setToken(v); }}
        review={{
          open: reviewOpen,
          pending: decisionView === 'pending' && wfStatus === 'awaiting_review',
          onToggle: () => setReviewOpen(o => !o),
        }}
      />

      <Rail
        workflowName={activeWorkflow}
        steps={wfSteps}
        currentStep={currentStep}
        gateStep={gateStep}
        wfStatus={wfStatus}
        decision={decisionView}
        timings={timings}
        file={file}
        datasetMeta={datasetMeta}
        fileInput={<input ref={fileRef} type="file" accept=".csv" hidden onChange={onFileChange} />}
        dropZone={
          <button
            type="button"
            className={`upload-zone ${drag ? 'drag' : ''}`}
            aria-labelledby="upload-zone-label"
            aria-describedby="upload-zone-hint"
            onDragOver={e => { e.preventDefault(); setDrag(true); }}
            onDragLeave={() => setDrag(false)}
            onDrop={onDrop}
            onClick={() => fileRef.current?.click()}
          >
            <Upload size={22} style={{ color: 'var(--text-dim)', marginBottom: 8 }} aria-hidden="true" />
            <span id="upload-zone-label" className="upload-label">
              <strong>Click to upload</strong> or drag & drop
            </span>
            <span id="upload-zone-hint" className="upload-hint">CSV · NONMEM-style (ID / TIME / DV / AMT columns)</span>
          </button>
        }
        canReplace={wfStatus === 'idle'}
        onReplace={() => fileRef.current?.click()}
        onToggleRoles={() => setShowRoles(s => !s)}
        rolesEnabled={hasData}
        workflows={[
          { key: 'nca_full', label: 'Run NCA workflow', primary: true },
          { key: 'poppk_modeling', label: 'Run modeling + engines',
            title: 'Fit structural models, then confirm across estimation engines (native FOCE-I + nlmixr2 + Monolix when installed), reviewed and QC-gated' },
          { key: 'poppk_full', label: 'Run full population PK',
            title: 'Full population PK: structural comparison (gated), NLME fit, SCM covariate build, residual diagnostics, covariate forest, VPC, adversarial review (gated), report' },
        ]}
        canRun={canRunWorkflow}
        running={loading && wfStatus === 'running' ? activeWorkflow : null}
        onRun={uploadAndRun}
        onNewSession={newSession}
      />

      <main className="main">
        {/* The human-review gate and the report card live in the review panel (S3). */}
        <div className="messages">
          <DocumentHeader
            title={docTitle}
            subtitle={docSubtitle}
            right={<ExportMenu disabled={loading || exportKinds.length === 0} items={exportItems} />}
          />
          {state?.dataset_metadata && (
            <ResultSection title="Dataset profile" agent="data_manager" tool="profile_pk_dataset"
              seal={profileSeal} sealing={!profileSeal}>
              <DatasetProfile items={profileItems} />
            </ResultSection>
          )}
          {messages.length === 0 && (
            <div className="empty">
              <FlaskConical size={36} />
              <span>Upload a PK dataset and choose a workflow.</span>
              <span style={{ fontSize: 12 }}>Or ask the agents directly.</span>
            </div>
          )}
          {messages.map(m => {
            // Read from the message's frozen snapshot when present (set at the
            // time the card was produced), else the live state.
            const st = m.snap ?? state;
            if (m.content === '__SPAGHETTI__' && st?.spaghetti_data) {
              return (
                <ResultSection key={m.id} {...sectionProps('__SPAGHETTI__', m)}>
                  <SpaghettiChart data={st.spaghetti_data as SpaghettiData} flagged={Object.keys(flaggedExtrap(st))} />
                </ResultSection>
              );
            }
            if (m.content === '__NCA_TABLE__' && st?.nca_summary) {
              return (
                <ResultSection key={m.id} {...sectionProps('__NCA_TABLE__', m)}>
                  <NcaTable state={st} />
                </ResultSection>
              );
            }
            if (m.content === '__NCA_LZ__' && st?.nca_plot_data) {
              return (
                <ResultSection key={m.id} {...sectionProps('__NCA_LZ__', m)}>
                  <NcaLzPlot data={st.nca_plot_data as NcaPlotData} sessionId={session?.id ?? ''}
                    flagged={flaggedExtrap(st)} hours={/^(h|hrs?|hours?)$/i.test(ncaUnits(st).time ?? '')} />
                </ResultSection>
              );
            }
            if (m.content === '__QC_CARD__' && st?.qc_verdict) {
              return (
                <ResultSection key={m.id} {...sectionProps('__QC_CARD__', m)}>
                  <QcCard state={st} onReviewLz={onReviewLz} onReviewInPanel={qc ? onReviewInPanel : undefined} />
                </ResultSection>
              );
            }
            if (m.content === '__BE__' && st?.be_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__BE__', m)}>
                  <BeCard r={st.be_results} />
                </ResultSection>
              );
            }
            if (m.content === '__DP__' && st?.dose_prop_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__DP__', m)}>
                  <DosePropCard r={st.dose_prop_results} />
                </ResultSection>
              );
            }
            if (m.content === '__CLINPHARM__' && st?.clinpharm_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__CLINPHARM__', m)} maxWidth={700}>
                  <ClinpharmCard r={st.clinpharm_results} />
                </ResultSection>
              );
            }
            if (m.content === '__STATS__' && st?.stats_advice) {
              return (
                <ResultSection key={m.id} {...sectionProps('__STATS__', m)}>
                  <StatsAdviceCard r={st.stats_advice} />
                </ResultSection>
              );
            }
            if (m.content === '__COMPARTMENTAL__' && st?.compartmental_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__COMPARTMENTAL__', m)}>
                  <CompartmentalCard r={st.compartmental_results} />
                </ResultSection>
              );
            }
            if (m.content === '__POPPK__' && st?.poppk_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__POPPK__', m)}>
                  <PopPkCard r={st.poppk_results} />
                </ResultSection>
              );
            }
            if (m.content === '__PENDING_TOOL__' && st?.pending_tool) {
              const p = st.pending_tool;
              // buttons only while THIS proposal is still the live one
              const live = !!state?.pending_tool
                && state.pending_tool.tool === p.tool
                && state.pending_tool.proposed_at === p.proposed_at;
              return (
                <ResultSection key={m.id} {...sectionProps('__PENDING_TOOL__', m)} maxWidth={640}>
                  <div className="gate-banner">
                    <AlertTriangle size={20} style={{ color: 'var(--yellow)', flexShrink: 0 }} />
                    <div className="gate-banner-text">
                      <div className="gate-banner-title">Run {p.tool}?</div>
                      <div className="gate-banner-sub">
                        Long-running fit (minutes to tens of minutes) — runs as a background job.
                        Nothing has been computed.
                      </div>
                      <pre style={{ fontSize: 11, margin: '6px 0 0', whiteSpace: 'pre-wrap' }}>
                        {JSON.stringify(p.args, null, 1)}
                      </pre>
                    </div>
                    {live ? (
                      <div className="gate-actions">
                        <button className="btn btn-green" disabled={loading} onClick={() => decidePendingTool(true)}>
                          <CheckCircle size={12} /> Approve
                        </button>
                        <button className="btn btn-red" disabled={loading} onClick={() => decidePendingTool(false)}>
                          <XCircle size={12} /> Reject
                        </button>
                      </div>
                    ) : (
                      <div className="gate-banner-sub">decided</div>
                    )}
                  </div>
                </ResultSection>
              );
            }
            if (m.content === '__PKMODEL__' && st?.pk_model_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__PKMODEL__', m)}>
                  <PkModelCard r={st.pk_model_results} />
                </ResultSection>
              );
            }
            if (m.content === '__VPC__' && st?.vpc_results) {
              const wide = !!(st.vpc_results.stratified || st.vpc_results.exposure_pc
                || st.vpc_results.blq_vpc);
              return (
                <ResultSection key={m.id} {...sectionProps('__VPC__', m)} maxWidth={wide ? 920 : 640}>
                  <VpcCard r={st.vpc_results} onRerun={runVpc} busy={loading}
                    covariates={vpcStrataOptions(st.dataset_metadata as
                      { columns?: { name: string }[]; detected_roles?: Record<string, string> } | null)} />
                </ResultSection>
              );
            }
            if (m.content === '__NLME__' && st?.nlme_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__NLME__', m)} maxWidth={640}>
                  <NlmeCard r={st.nlme_results} />
                </ResultSection>
              );
            }
            if (m.content === '__PRIORCHECK__' && st?.prior_check_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__PRIORCHECK__', m)} maxWidth={640}>
                  <PriorCheckCard r={st.prior_check_results} />
                </ResultSection>
              );
            }
            if (m.content === '__SCM__' && st?.scm_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__SCM__', m)} maxWidth={680}>
                  <ScmCard r={st.scm_results} />
                </ResultSection>
              );
            }
            if (m.content === '__ENGINES__' && st?.engine_comparison_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__ENGINES__', m)} maxWidth={700}>
                  <EngineComparisonCard r={st.engine_comparison_results} />
                </ResultSection>
              );
            }
            if (m.content === '__FORECAST__' && st?.forecast_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__FORECAST__', m)} maxWidth={560}>
                  <ForecastCard r={st.forecast_results} />
                </ResultSection>
              );
            }
            if (m.content === '__DIAG__' && st?.diagnostics_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__DIAG__', m)} maxWidth={660}>
                  <DiagnosticsCard r={st.diagnostics_results} />
                </ResultSection>
              );
            }
            if (m.content === '__FOREST__' && st?.forest_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__FOREST__', m)} maxWidth={660}>
                  <ForestCard r={st.forest_results} />
                </ResultSection>
              );
            }
            if (m.content === '__SIMEST__' && st?.simest_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__SIMEST__', m)} maxWidth={660}>
                  <SimestCard r={st.simest_results} />
                </ResultSection>
              );
            }
            if (m.content === '__BOOTSTRAP__' && st?.bootstrap_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__BOOTSTRAP__', m)} maxWidth={760}>
                  <BootstrapCard r={st.bootstrap_results} />
                </ResultSection>
              );
            }
            if (m.content === '__SIR__' && st?.sir_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__SIR__', m)} maxWidth={700}>
                  <SirCard r={st.sir_results} />
                </ResultSection>
              );
            }
            if (m.content === '__PROFILE__' && st?.profile_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__PROFILE__', m)} maxWidth={760}>
                  <ProfileCard r={st.profile_results} />
                </ResultSection>
              );
            }
            if (m.content === '__SWEEP__' && st?.dose_sweep_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__SWEEP__', m)} maxWidth={640}>
                  <DoseSweepCard r={st.dose_sweep_results} />
                </ResultSection>
              );
            }
            if (m.content === '__CLINSIM__' && st?.clinsim_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__CLINSIM__', m)} maxWidth={660}>
                  <ClinsimCard r={st.clinsim_results} onRerun={runClinsim} busy={loading} />
                </ResultSection>
              );
            }
            if (m.content === '__EXPFOREST__' && st?.exposure_forest_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__EXPFOREST__', m)} maxWidth={680}>
                  <ExposureForestCard r={st.exposure_forest_results} />
                </ResultSection>
              );
            }
            if (m.content === '__SPECIALPOP__' && st?.special_pop_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__SPECIALPOP__', m)} maxWidth={720}>
                  <SpecialPopCard r={st.special_pop_results} onRerun={runSpecialPop} busy={loading} />
                </ResultSection>
              );
            }
            if (m.content === '__INDIVEXP__' && st?.individual_exposures) {
              return (
                <ResultSection key={m.id} {...sectionProps('__INDIVEXP__', m)} maxWidth={640}>
                  <IndividualExposuresCard r={st.individual_exposures} />
                </ResultSection>
              );
            }
            if (m.content === '__PEDIATRIC__' && st?.pediatric_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__PEDIATRIC__', m)} maxWidth={760}>
                  <PediatricCard r={st.pediatric_results} onRerun={runPediatric} busy={loading} />
                </ResultSection>
              );
            }
            if (m.content === '__SIM__' && st?.simulation_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__SIM__', m)} maxWidth={640}>
                  <SimChart sim={st.simulation_results} />
                </ResultSection>
              );
            }
            if (m.content === '__REVIEW__' && st?.review_results) {
              return (
                <ResultSection key={m.id} {...sectionProps('__REVIEW__', m)} maxWidth={720}>
                  <ReviewCard r={st.review_results} />
                </ResultSection>
              );
            }
            if (m.content === '__REPORT__') return null;
            return <MessageBubble key={m.id} msg={m} agent={m.agent} />;
          })}
          {/* Executed steps that put no section above (load, validate, review…), folded. */}
          {runLog.length > 0 && (
            <section className="section run-log">
              <details>
                <summary>
                  <ChevronRight size={14} className="run-log-chev" aria-hidden="true" />
                  <h2>Run log</h2>
                  <span className="section-attr">
                    {runLog.length} step{runLog.length === 1 ? '' : 's'} without a result section
                  </span>
                </summary>
                <ol className="run-log-list">
                  {runLog.map((s, i) => (
                    <li key={`${s.step}-${i}`}>
                      <div className="run-log-head">
                        <span className="run-log-label">{s.label}</span>
                        <span className="section-attr">{agentLabel(s.agent)} · <code>{s.tool}</code></span>
                      </div>
                      <div className="run-log-summary">{stripLabelPrefix(s.label, s.summary)}</div>
                    </li>
                  ))}
                </ol>
              </details>
            </section>
          )}
          {/* Panels toggled from the Actions menu render as document sections (no attribution). */}
          {showRoles && state && (
            <ResultSection title="Column roles">
              <RolesEditor state={state} onApply={applyRoles} loading={loading} />
            </ResultSection>
          )}
          {showFlexplot && session && (
            <ResultSection title="Flexplot">
              <FlexplotPanel sessionId={session.id} />
            </ResultSection>
          )}
          {showSkills && (
            <ResultSection title="Skills">
              <SkillsPanel skills={skills} loading={loading}
                datasetPath={state?.dataset_path ?? null}
                onRun={runSkill} onDelete={deleteSkill} onMarkdown={n => api.skillMarkdown(n)} />
            </ResultSection>
          )}
          {calcOpen && session && (
            <ResultSection title="Clinical pharmacology calculators">
              <CalculatorsPanel sessionId={session.id} busy={loading} onResult={(st, summary) => {
                setState(st);
                pushMsg({ role: 'assistant', content: summary, agent: 'clinpharm', id: '' });
                pushMsg({ role: 'assistant', content: '__CLINPHARM__', agent: 'clinpharm', id: '', snap: st });
              }} />
            </ResultSection>
          )}
          <div ref={messagesEnd} />
        </div>

        <Composer
          value={input}
          onChange={setInput}
          onKeyDown={onKeyDown}
          onSend={() => sendChat()}
          textDisabled={!session || loading}
          sendDisabled={!input.trim() || !session || loading}
          loading={loading}
          jobNote={jobNote}
          actionsOpen={actionsOpen}
          onToggleActions={() => setActionsOpen(o => !o)}
          actionsRef={actionsRef}
        >
          <ActionsMenu open={actionsOpen} onClose={() => setActionsOpen(false)} anchorRef={actionsRef}>
            {hasData && (
              <div className="quick-actions">
                <span className="quick-actions-label">Run on this data</span>
                {QUICK_ACTIONS.map(qa => (
                  <button
                    key={qa.label}
                    className="chip"
                    disabled={loading}
                    onClick={() => sendChat(qa.msg)}
                  >
                    {qa.label}
                  </button>
                ))}
              </div>
            )}

            {hasData && (
              <div className="quick-actions">
                <span className="quick-actions-label">Data</span>
                <button className="chip" data-keep-open disabled={loading} onClick={() => setShowRoles(s => !s)}>
                  {showRoles ? 'Hide columns' : 'Columns / roles'}
                </button>
              </div>
            )}

            {hasData && (
              <div className="quick-actions">
                <span className="quick-actions-label">Visualize</span>
                <button className="chip" data-keep-open disabled={loading} onClick={() => setShowFlexplot(s => !s)}>
                  {showFlexplot ? 'Hide flexplot' : 'Flexplot'}
                </button>
              </div>
            )}

            {hasData && pkModels.length > 0 && (
              <div className="quick-actions">
                <span className="quick-actions-label">PK model library</span>
                <select
                  className="model-select"
                  aria-label="PK model"
                  value={selectedModel}
                  onChange={e => setSelectedModel(e.target.value)}
                  disabled={loading}
                >
                  {['IV linear', 'Oral', 'Nonlinear', 'PK/PD'].map(group => (
                    <optgroup key={group} label={group}>
                      {pkModels.filter(m => m.group === group).map(m => (
                        <option key={m.key} value={m.key}>{m.label}{m.has_pd ? ' (needs PD)' : ''}</option>
                      ))}
                    </optgroup>
                  ))}
                </select>
                <button className="chip" disabled={loading} onClick={() => runPkModel({ model_key: selectedModel })}>
                  Fit model
                </button>
                <button className="chip" disabled={loading} onClick={() => runPkModel({ compare: true })}>
                  Compare oral models
                </button>
              </div>
            )}

            {state?.pk_model_results?.status === 'ok' && (
              <div className="quick-actions">
                <span className="quick-actions-label">Forecast</span>
                <label className="sim-field">dose
                  <input type="number" value={simDose} disabled={loading}
                    onChange={e => setSimDose(Number(e.target.value))} />
                </label>
                <label className="sim-field">q (h)
                  <input type="number" value={simTau} disabled={loading}
                    onChange={e => setSimTau(Number(e.target.value))} />
                </label>
                <label className="sim-field"># doses
                  <input type="number" value={simNDoses} disabled={loading}
                    onChange={e => setSimNDoses(Number(e.target.value))} />
                </label>
                <label className="sim-field">to (h)
                  <input type="number" placeholder="auto" value={simTmax} disabled={loading}
                    onChange={e => setSimTmax(e.target.value === '' ? '' : Number(e.target.value))} />
                </label>
                <button className="chip" disabled={loading} onClick={runSimulate}>Simulate forward</button>
              </div>
            )}

            {state?.pk_model_results?.status === 'ok' && (
              <div className="quick-actions">
                <span className="quick-actions-label">Population (NLME)</span>
                <button className="chip" disabled={loading} onClick={() => runNlme('focei')}
                  title="Single cold start — fastest and fully reproducible. On harder models (several IIV terms, covariates) a cold start can converge to the wrong optimum while still reporting success.">
                  FOCE-I only
                </button>
                <button className="chip" disabled={loading} onClick={() => runNlme('saem')}
                  title="Stochastic EM — explores rather than descends, so it is far less sensitive to starting values, but gives no exact Laplace OFV or asymptotic standard errors.">
                  SAEM
                </button>
                <button className="chip" disabled={loading} onClick={() => runNlme('auto')}
                  title="Runs FOCE-I, then probes with an independent SAEM-seeded start. If the two agree it stops there; if they disagree it escalates to a multi-start search and returns the lowest-OFV fit. Never worse than FOCE-I alone, but much slower whenever it escalates.">
                  Auto
                </button>
                <label className="sim-field">error
                  <select className="model-select" style={{ maxWidth: 130 }} value={errorModel}
                    disabled={loading} onChange={e => setErrorModel(e.target.value)}>
                    <option value="proportional">proportional</option>
                    <option value="additive">additive</option>
                    <option value="combined">combined</option>
                  </select>
                </label>
                <button className="chip" disabled={loading || !state?.nlme_results}
                  onClick={() => runNlme('focei', { prior_from: 'nlme' })}
                  title="MAP fit with the current fit as an informative prior (Bayesian borrowing): informative prior from the stored fit's covariance. Fit adults first, then load the sparse (e.g. pediatric) data and click this.">
                  MAP (informative prior)
                </button>
                <button className="chip" disabled={loading || !state?.nlme_results}
                  onClick={() => runNlme('focei', { prior_from: 'nlme', prior_var: 1.0 })}
                  title="MAP fit with a WEAKLY informative prior (variance 1.0 on all params) — the estimate follows the data more than the prior.">
                  MAP (weak prior)
                </button>
                <button className="chip" disabled={loading || !(state?.nlme_results && state.nlme_results.map)}
                  onClick={runPriorCheck}
                  title="Prior-predictive band vs the data + prior-vs-posterior shrinkage. Needs a MAP fit.">
                  Prior check
                </button>
                <button className="chip" disabled={loading} onClick={runScm}
                  title="Stepwise covariate modeling: forward selection (p<0.05) + backward elimination (p<0.01) over dataset covariates">
                  Covariate SCM
                </button>
                <button className="chip" disabled={loading} onClick={runEngineComparison}
                  title="Fit the model across estimation engines (native FOCE-I + nlmixr2 + Monolix if installed); winner chosen by prediction accuracy, not cross-engine OFV">
                  Compare engines
                </button>
              </div>
            )}

            {state?.nlme_results?.status === 'ok' && (
              <div className="quick-actions">
                <span className="quick-actions-label">TDM / MAP forecast</span>
                <label className="sim-field">dose
                  <input type="number" value={fcDose} disabled={loading}
                    onChange={e => setFcDose(Number(e.target.value))} />
                </label>
                <label className="sim-field">q (h)
                  <input type="number" value={fcTau} disabled={loading}
                    onChange={e => setFcTau(Number(e.target.value))} />
                </label>
                <label className="sim-field">levels (t,conc; …)
                  <input type="text" style={{ width: 150 }} placeholder="e.g. 48.5,1.2; 72,0.4"
                    value={fcLevels} disabled={loading} onChange={e => setFcLevels(e.target.value)} />
                </label>
                <label className="sim-field">target
                  <input type="text" style={{ width: 60 }} placeholder="opt." value={fcTarget}
                    disabled={loading} onChange={e => setFcTarget(e.target.value)} />
                </label>
                <select className="model-select" style={{ maxWidth: 90 }} value={fcMetric}
                  disabled={loading} aria-label="Forecast target metric" onChange={e => setFcMetric(e.target.value)}>
                  <option value="cmin">Cmin</option>
                  <option value="cmax">Cmax</option>
                  <option value="cavg">Cavg</option>
                  <option value="auc_tau">AUCτ</option>
                </select>
                <button className="chip" disabled={loading} onClick={runForecast}
                  title="MAP/empirical-Bayes individualization from the fitted population model + measured levels">
                  MAP forecast
                </button>
              </div>
            )}

            {state?.nlme_results?.status === 'ok' && !(state.nlme_results.covariate_effects?.length) && (
              <div className="quick-actions">
                <span className="quick-actions-label">Trial-design precision check</span>
                {!seShowConfirm ? (
                  <button className="chip" data-keep-open disabled={loading} onClick={() => setSeShowConfirm(true)}
                    title="Simulate replicate trials under a proposed design and re-fit each — checks whether the 95% CI lands within 60-140% of its own estimate (up to 10 replicates; runs several real NLME fits, several minutes)">
                    Simulation-estimation…
                  </button>
                ) : (
                  <>
                    <label className="sim-field">N subjects
                      <input type="number" value={seN} disabled={loading}
                        onChange={e => setSeN(Number(e.target.value))} />
                    </label>
                    <label className="sim-field">sample times (h)
                      <input type="text" style={{ width: 160 }} value={seObsT} disabled={loading}
                        onChange={e => setSeObsT(e.target.value)} />
                    </label>
                    <label className="sim-field">dose
                      <input type="number" value={seDose} disabled={loading}
                        onChange={e => setSeDose(Number(e.target.value))} />
                    </label>
                    <label className="sim-field">replicates (≤10)
                      <input type="number" min={1} max={10} value={seNRep} disabled={loading}
                        onChange={e => setSeNRep(Math.max(1, Math.min(10, Number(e.target.value))))} />
                    </label>
                    <button className="chip" disabled={loading} onClick={runSimest}
                      title="Confirms and runs — several real NLME fits, several minutes to tens of minutes; holds this session while running">
                      Confirm &amp; run
                    </button>
                    <button className="chip" data-keep-open disabled={loading} onClick={() => setSeShowConfirm(false)}>
                      Cancel
                    </button>
                  </>
                )}
              </div>
            )}

            {state?.nlme_results?.status === 'ok' && !!(state.nlme_results.covariate_effects?.length) && (
              <div className="quick-actions">
                <span className="quick-actions-label" style={{ color: 'var(--text-dim)' }}>
                  Trial-design precision check unavailable: not supported for models with covariate effects.
                </span>
              </div>
            )}

            {state?.pk_model_results?.status === 'ok' && (
              <div className="quick-actions">
                <span className="quick-actions-label">Diagnostics</span>
                <button className="chip" disabled={loading} onClick={() => runVpc()}>VPC / goodness-of-fit</button>
                <button className="chip" disabled={loading} onClick={runDiagnostics}>Residual diagnostics</button>
                <button className="chip" disabled={loading} onClick={runCovariateForest}
                  title="Covariate GMR forest plot from a converged run_nlme or run_scm covariate model">
                  Covariate forest
                </button>
                <button className="chip" disabled={loading} onClick={runExposureForest}
                  title="Simulated exposure forest: relative AUC/Cmax across covariate extremes with the 0.8–1.25 band">
                  Exposure forest
                </button>
                <button className="chip" disabled={loading} onClick={() => runSpecialPop()}
                  title="Special-population simulation: steady-state exposure by renal function (or covariate) vs the normal reference band → dose adjustment">
                  Special populations
                </button>
                <button className="chip" disabled={loading} onClick={runIndividualExposures}
                  title="Per-subject steady-state AUCss/Cmax,ss from the fitted EBEs (needs an NLME fit)">
                  Individual exposures
                </button>
                <button className="chip" disabled={loading} onClick={() => runPediatric()}
                  title="Pediatric dose-finding: age×weight exposure vs the adult range → the dose matching adult exposure (supports estimated allometry)">
                  Pediatric doses
                </button>
                <label className="sim-field">doses
                  <input type="text" style={{ width: 130 }} placeholder="e.g. 2500,5000,10000"
                    value={sweepDoses} disabled={loading}
                    onChange={e => setSweepDoses(e.target.value)} />
                </label>
                <button className="chip" disabled={loading} onClick={runDoseSweep}>Dose sweep</button>
                <button className="chip" disabled={loading} onClick={() => runClinsim()}
                  title="Clinical trial simulation: virtual population across a dose grid → probability of target attainment + dose recommendation">
                  Trial simulation (PTA)</button>
                <span className="quick-actions-note">Dose sweep uses the Forecast q / # doses / to values.</span>
              </div>
            )}

            {(state?.nca_parameters?.length || state?.nlme_results?.status === 'ok') && (
              <div className="quick-actions">
                <span className="quick-actions-label">Review &amp; skills</span>
                <button className="chip" disabled={loading} onClick={runReview}
                  title="Adversarial reviewer: independently recompute and challenge every result; loop to a checkable goal">
                  Adversarial review
                </button>
                <button className="chip" disabled={loading} onClick={captureSkill}
                  title="Capture this session's analysis sequence as a reusable, replayable skill">
                  Capture as skill
                </button>
                <button className="chip" data-keep-open disabled={loading}
                  onClick={() => { setShowSkills(s => !s); if (!showSkills) refreshSkills(); }}>
                  {showSkills ? 'Hide skills' : `Skills${skills.length ? ` (${skills.length})` : ''}`}
                </button>
              </div>
            )}

            <div className="quick-actions">
              <span className="quick-actions-label">Calculators</span>
              <button className="chip" data-keep-open onClick={() => setCalcOpen(o => !o)}>{calcOpen ? 'Hide calculators' : 'Clinical pharmacology calculators'}</button>
            </div>
          </ActionsMenu>
        </Composer>
      </main>

      <ReviewPanel
        open={reviewOpen}
        onClose={closeReview}
        wfStatus={wfStatus}
        decision={decisionView}
        deciding={deciding}
        loading={loading}
        jobNote={jobNote}
        gate={gate}
        signer={gateSigner}
        onApprove={r => resume(true, r)}
        onReject={r => resume(false, r)}
        qc={qc}
        evidence={evidence}
        outcome={outcome}
        audit={audit}
        integrity={auditIntegrity}
        onVerify={verifyChain}
        verifyNote={verifyNote}
        onReviewLz={onReviewLz}
      />
    </>
  );
}
