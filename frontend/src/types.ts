export interface Session {
  id: string;
  created_at: string;
}

export interface JobResult {
  agent?: string;
  tool?: string;
  summary: string;
  state: PharmState;
  result?: unknown;
  audit_ok?: boolean;
}

export interface SpaghettiSeries {
  id: string;
  x: number[];
  y: number[];
  blq_x?: number[];
  dose?: number | null;
}
export interface SpaghettiData {
  series: SpaghettiSeries[];
  log_scale: boolean;
  n_subjects: number;
  blq_excluded: number;
  x_label: string;
  y_label: string;
}

export interface LzSubject {
  id: string;
  x: number[]; y: number[];
  lz_x: number[]; lz_y: number[];
  fit_x: number[]; fit_y: number[];
  lambda_z: number | null;
  t_half: number | null;
  r2_adj: number | null;
  n_pts: number | null;
  Tmax: number | null;
}
export interface NcaPlotData {
  subjects: LzSubject[];
}

export interface EngineResultRow {
  engine: string;
  model_name: string;
  converged: boolean;
  runtime_s: number | null;
  ofv: number | null;
  aic: number | null;
  bic: number | null;
  params: Record<string, number>;
  pred_rmse: number | null;
  pred_bias: number | null;
  pred_r2: number | null;
  vpc_coverage90: number | null;
  n_map_fallback?: number;
  status: string;
  message?: string;
}

export interface EngineLikelihoodRow {
  model: string;
  ofv: number | null;
  aic: number | null;
  bic: number | null;
}

export interface EngineComparisonResults {
  status: string;
  message?: string;
  winner: EngineResultRow | null;
  prediction_ranking: EngineResultRow[];
  within_engine_likelihood: Record<string, EngineLikelihoodRow[]>;
  results?: EngineResultRow[];
  selection_metric?: string;
  note?: string;
  n_engines?: number;
  n_available?: number;
  n_candidates?: number;
}

export interface PharmState {
  session_id: string;
  dataset_id: string | null;
  dataset_path: string | null;
  dataset_metadata: Record<string, unknown> | null;
  data_quality: Record<string, unknown> | null;
  spaghetti_data: SpaghettiData | null;
  nca_parameters: NcaSubject[] | null;
  nca_summary: NcaSummary | null;
  nca_plot_data: NcaPlotData | null;
  be_results: BeResults | null;
  dose_prop_results: DosePropResults | null;
  compartmental_results: CompartmentalResults | null;
  poppk_results: PopPkResults | null;
  pk_model_results: PkModelResults | null;
  nlme_results: NlmeResults | null;
  scm_results: ScmResults | null;
  forecast_results: ForecastResults | null;
  simulation_results: SimulationResults | null;
  vpc_results: VpcResults | null;
  diagnostics_results: DiagnosticsResults | null;
  forest_results: ForestResults | null;
  engine_comparison_results: EngineComparisonResults | null;
  dose_sweep_results: DoseSweepResults | null;
  simest_results: SimestResults | null;
  qc_verdict: string | null;
  qc_issues: QcIssue[] | null;
  qc_checklist: QcCheck[] | null;
  review_results: ReviewResults | null;
  regulatory_report_path: string | null;
  report_path: string | null;
  workflow_name: string | null;
  current_step: number | null;
  last_agent: string | null;
}

export type Severity = 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';

export interface ReviewFinding {
  id: string;
  severity: Severity;
  target: string;
  claim: string;
  evidence: string;
  suggested_action: string;
  resolved: boolean;
}

export interface ReviewResults {
  goal: string;
  goal_met: boolean;
  findings: ReviewFinding[];
  counts: Record<Severity, number>;
  n_findings: number;
  checked?: Record<string, boolean>;
}

export interface ReviewLoopResult {
  goal: string;
  goal_met: boolean;
  iterations: number;
  findings: ReviewFinding[];
  counts: Record<Severity, number>;
  state: PharmState;
  audit_ok: boolean;
}

export interface SkillStep {
  agent: string;
  tool: string;
  args: Record<string, unknown>;
}

export interface SkillDef {
  name: string;
  description: string;
  goal: string;
  steps: SkillStep[];
  source_session: string | null;
  owner: string | null;
  created_at: string;
  version: number;
}

export interface BeParam {
  gmr_pct: number | null;
  ci_lower_pct: number | null;
  ci_upper_pct: number | null;
  within_limits: boolean;
  cv_intra_pct: number | null;
}

export interface BeResults {
  status: string;
  design?: string;
  test_level?: string;
  reference_level?: string;
  n_test?: number;
  n_reference?: number;
  bioequivalent?: boolean;
  limits?: number[];
  parameters?: Record<string, BeParam>;
  message?: string;
}

export interface DpParam {
  slope: number | null;
  slope_ci_lower: number | null;
  slope_ci_upper: number | null;
  r_squared: number | null;
  dose_ratio: number | null;
  critical_region: number[] | null;
  proportional: boolean | null;
}

export interface DosePropResults {
  status: string;
  proportional?: boolean | null;
  dose_levels?: number[];
  parameters?: Record<string, DpParam>;
  message?: string;
}

export interface CompFit {
  subject: string | number;
  model: string;
  converged: boolean;
  params: Record<string, number> | null;
  aic: number | null;
  r_squared: number | null;
}

export interface CompartmentalResults {
  status?: string;
  steady_state?: boolean;
  n_subjects: number;
  n_converged: number;
  model_selection_counts: Record<string, number>;
  fits: CompFit[];
}

export interface PopPkParam {
  typical_value: number | null;
  iiv_cv_pct: number | null;
  median: number | null;
  n: number;
}

export interface PopPkResults {
  status: string;
  source?: string;
  method?: string;
  n_subjects?: number;
  parameters?: Record<string, PopPkParam>;
  covariate_screen?: Record<string, unknown>;
  message?: string;
}

export interface PkModelDef {
  key: string;
  label: string;
  group: string;
  is_iv: boolean;
  has_pd: boolean;
  params: string[];
}

export interface PkPopParam {
  typical_value: number | null;
  iiv_cv_pct: number | null;
  median: number | null;
  n: number;
}

export interface PkIndivFit {
  subject: string | number;
  converged: boolean;
  params: Record<string, number> | null;
  aic: number | null;
  r_squared: number | null;
}

export interface PkModelFit {
  model_key: string;
  label: string;
  group: string;
  n_subjects: number;
  n_converged: number;
  mean_aic: number | null;
  total_aic: number | null;
  population: { method: string; n_subjects: number; parameters: Record<string, PkPopParam> };
  individual_fits: PkIndivFit[];
}

export interface PkRankRow {
  model_key: string;
  label: string;
  group: string;
  n_converged: number;
  n_subjects: number;
  total_aic: number | null;
  mean_aic: number | null;
}

export interface NlmeResults {
  status: string;
  method?: string;
  model_key?: string;
  label?: string;
  message?: string;
  iiv_params?: string[];
  error_model?: string;
  theta?: Record<string, number>;
  theta_rse_pct?: Record<string, number>;
  omega_cv_pct?: Record<string, number>;
  omega_rse_pct?: Record<string, number>;
  sigma?: { prop: number | null; add: number | null };
  sigma_rse_pct?: { prop: number | null; add: number | null };
  covariate_effects?: CovariateEffect[];
  /** Present only for method="auto": which starts were tried and why. */
  auto?: {
    escalated: boolean;
    reason: string;
    tol: number;
    n_candidates: number;
    winner: string;
    candidate_ofv: Record<string, number | null>;
  };
  /** Present only for method="focei_saem": the SAEM burn-in that seeded the fit. */
  seeded_by?: { method: string; iterations: number; ofv: number; converged: boolean } | null;
  ofv?: number;
  condition_number?: number | null;
  cov_note?: string;
  shrinkage_pct?: Record<string, number>;
  n_subjects?: number;
  n_obs?: number;
  n_blq?: number;
  converged?: boolean;
  iterations?: number;
}

export interface CovariateEffect {
  param: string;
  covariate: string;
  kind: string;
  center?: number | null;
  levels?: string[] | null;
  coefficient: number | Record<string, number>;
  rse_pct?: number | Record<string, number> | null;
  description: string;
}

export interface ScmStep {
  phase: string;
  effect: string;
  delta_ofv: number;
  df: number;
  crit: number;
  p: number;
  ofv: number;
  decision: string;
}

export interface ScmResults {
  status: string;
  message?: string;
  model_key?: string;
  label?: string;
  base_ofv?: number | null;
  final_ofv?: number | null;
  forward_p?: number;
  backward_p?: number;
  selected?: { param: string; covariate: string; kind: string }[];
  steps?: ScmStep[];
  n_candidates?: number;
  final?: NlmeResults;
  note?: string;
}

export interface ForecastResults {
  status: string;
  message?: string;
  model_key?: string;
  label?: string;
  eta?: Record<string, number>;
  n_obs?: number;
  individual_params?: Record<string, number>;
  typical_params?: Record<string, number>;
  dose?: number;
  tau?: number;
  wt?: number;
  measured?: { time: number; conc: number }[];
  ss_individual?: Record<string, number>;
  ss_population?: Record<string, number>;
  forecast?: { times: number[]; individual: number[]; population: number[] };
  recommendation?: {
    target_metric: string;
    target: number;
    recommended_dose: number | null;
    predicted?: Record<string, number>;
    note?: string;
  };
}

export interface DiagnosticsResults {
  status: string;
  model_key?: string;
  label?: string;
  message?: string;
  residuals?: {
    time: number[]; obs: number[]; ipred: number[]; pred: number[];
    iwres: number[]; iwres_std: number[]; tad: (number | null)[];
    summary: { n: number; iwres_mean: number | null; iwres_sd: number | null; n_tad_null?: number };
  };
  npde?: {
    status?: 'needs_nlme' | 'blq_unsupported';
    message?: string;
    n_blq?: number;
    metric?: 'npd';
    time?: number[]; pred?: number[]; npde?: number[]; tad?: (number | null)[];
    summary?: {
      n: number; mean: number | null; sd: number | null; pct_outside_1_96: number | null;
      n_tad_null?: number; sigma_prop?: number; sigma_add?: number;
    };
  };
  cwres?: {
    status?: 'needs_nlme';
    message?: string;
    time?: number[]; obs?: number[]; ipred?: number[]; cpred?: number[];
    cwres?: number[]; iwres?: number[]; tad?: (number | null)[];
    skipped_subjects?: string[]; cov_fallback_subjects?: string[];
    summary?: {
      n: number; n_subjects_used?: number; n_subjects_skipped?: number;
      n_blq_dropped?: number; n_floored_dropped?: number; n_tad_null?: number;
      cov_fallback_n?: number; n_etas_reused?: number; n_etas_resolved?: number;
      cwres_mean: number | null; cwres_sd: number | null;
      cwres_pct_outside_1_96: number | null; eps_shrinkage_pct?: number | null;
      interaction?: boolean; cwres_variant?: 'focei' | 'foce';
    };
  };
  nlme_provenance?: string | null;
}

export interface ForestRow {
  param: string;
  covariate: string;
  kind: 'power' | 'linear' | 'exponential' | 'categorical';
  eval_label: string;
  eval_value: number | string | null;
  gmr: number | null;
  ci_lo: number | null;
  ci_hi: number | null;
  ci_source: 'wald_loglinear' | 'delta_nonlinear' | 'undefined_extrapolation' | 'unavailable' | 'reference';
  omega_cv_pct: number | null;
  allometric_note: boolean;
  outside_reference_band: boolean;
}

export interface ForestResults {
  status: string;
  message?: string;
  model_key?: string;
  label?: string;
  source?: 'nlme' | 'scm';
  percentiles?: number[];
  rows?: ForestRow[];
  x_range?: [number, number] | null;
  bounds?: [number, number] | null;
  ci_level?: number;
  notes?: string[];
  summary?: { n_rows: number; n_effects: number };
  cov_stats?: Record<string, { n_cov: number; cov_min?: number; cov_max?: number; levels?: string[] }>;
}

export interface SimestPerParam {
  truth: number | null;
  gm_point_estimate: number | null;
  rel_bias_pct: number | null;
  rmse_pct: number | null;
  cv_across_replicates_pct: number | null;
  n_pass_precision_criterion: number;
  pct_within_60_140_of_own_estimate: number | null;
  pct_within_60_140_strict: number | null;
  coverage_wilson_ci_pct: [number | null, number | null];
  mean_theta_se_log: number | null;
}

export interface SimestReplicate {
  theta: Record<string, number>;
  ci: Record<string, [number, number]> | null;
}

export interface SimestResults {
  status: string;
  message?: string;
  model_key?: string;
  params?: string[];
  rse_convention?: 'log_scale_se';
  citation?: string;
  design_limitations?: string[];
  n_rep_requested?: number;
  n_rep_planned?: number;
  n_rep_completed?: number;
  n_point_evaluable?: number;
  n_ci_evaluable?: number;
  n_excluded?: number;
  excluded_reasons?: Record<string, number>;
  n_resampled_total?: number;
  n_negative_draws_total?: number;
  ci_validity?: 'assessable' | 'unassessable';
  criterion?: {
    pct_within_60_140_strict: number | null;
    pct_within_60_140_of_own_estimate: number | null;
    target_pct: number | null;
    criterion_met: boolean | null;
  };
  per_param?: Record<string, SimestPerParam>;
  replicates?: SimestReplicate[];
  est_minutes_rough?: number | null;
  elapsed_seconds?: number;
}

export interface VpcResults {
  status: string;
  model_key?: string;
  label?: string;
  message?: string;
  gof?: { r2_log_ipred: number | null; rmse_log_ipred: number | null; n: number };
  obs_vs_pred?: { observed: number[]; ipred: number[]; pred: number[] };
  vpc?: { times: number[]; p05: number[]; p50: number[]; p95: number[] };
  vpc_dose?: number | null;
  obs_t?: number[];
  obs_c?: number[];
  pcvpc?: PcVpcResult;
  /** Present only when run_vpc was called with stratify_by / dose_normalize / x_by. */
  stratified?: {
    status: string;
    stratify_by?: string | null;
    kind?: string;
    correction?: 'pred' | 'dose' | 'none';
    x_by?: 'time' | 'tad';
    message?: string;
    available?: string[];
    strata?: { label: string; n: number; bins: PcVpcBin[] }[];
    skipped?: { label: string; n: number; reason: string }[];
  };
  /** Present only when run_vpc was called with exposure_check. */
  exposure_pc?: ExposurePc;
  /** Present only when run_vpc was called with blq_check. */
  blq_vpc?: BlqVpc;
}

export interface ExposureMetric {
  observed: number;
  sim_median: number;
  sim_lo: number;
  sim_hi: number;
  within: boolean;
  hist: { edges: number[]; counts: number[] };
}

export interface ExposurePc {
  status: string;
  group_by?: string | null;
  kind?: string;
  n_sim?: number;
  ci?: number[];
  message?: string;
  multiple_dose?: boolean;
  groups?: { label: string; n: number; auc: ExposureMetric; cmax: ExposureMetric }[];
  skipped?: { label: string; n: number; reason: string }[];
}

export interface BlqBin {
  x: number | null;
  n: number;
  obs_frac: number | null;
  sim_med: number | null;
  sim_lo: number | null;
  sim_hi: number | null;
}

export interface BlqVpc {
  status: string;
  n_bins?: number;
  n_sim?: number;
  lloq?: number;
  x_by?: 'time' | 'tad';
  n_blq?: number;
  message?: string;
  bins?: BlqBin[];
}

export interface PcVpcBin {
  t: number | null; n: number;
  obs_p05: number | null; obs_p50: number | null; obs_p95: number | null;
  sim_p05: number | null; sim_p50: number | null; sim_p95: number | null;
  sim_med_lo: number | null; sim_med_hi: number | null;
}

export interface PcVpcResult {
  status: string;
  n_bins: number;
  n_sim: number;
  correction?: 'pred' | 'dose' | 'none';
  x_by?: 'time' | 'tad';
  bins: PcVpcBin[];
}

export interface DoseProfile {
  dose: number;
  times: number[];
  cp: number[];
  eff?: number[];
  cmax: number;
  auc_tau: number;
  cavg: number;
  ctrough: number;
}

export interface DoseSweepResults {
  status: string;
  model_key?: string;
  label?: string;
  message?: string;
  tau?: number;
  n_doses?: number;
  tmax?: number;
  profiles?: DoseProfile[];
}

export interface SimulationResults {
  status: string;
  model_key?: string;
  label?: string;
  has_pd?: boolean;
  from_fit?: boolean;
  params?: Record<string, number>;
  regimen?: { dose: number; tau: number; n_doses: number; tmax: number; wt: number; rate: number };
  times?: number[];
  cp?: number[];
  eff?: number[];
  cmax?: number | null;
  message?: string;
}

export interface PkModelResults {
  status: string;
  mode?: 'fit' | 'compare';
  multiple_dose?: boolean;
  is_pkpd?: boolean;
  message?: string;
  // fit mode (flattened PkModelFit fields)
  model_key?: string;
  label?: string;
  n_subjects?: number;
  n_converged?: number;
  mean_aic?: number | null;
  total_aic?: number | null;
  population?: { method: string; n_subjects: number; parameters: Record<string, PkPopParam> };
  individual_fits?: PkIndivFit[];
  // compare mode
  ranking?: PkRankRow[];
  best_model?: string | null;
  best?: PkModelFit | null;
}

export interface NcaSubject {
  subject: string | number;
  dose: number;
  Cmax: number;
  Tmax: number;
  AUC_last: number;
  AUC_inf: number | null;
  t_half: number;
  CL_F: number;
  Vz_F: number;
  lambda_z: number;
  pct_AUC_extrap: number | null;
  // steady-state extras (present when nca_summary.steady_state)
  steady_state?: boolean;
  tau?: number;
  Cmin?: number;
  Cavg?: number;
  AUC_tau?: number;
  fluctuation_pct?: number | null;
  accumulation_ratio?: number | null;
}

export interface NcaSummary {
  n_subjects: number;
  by_dose: DoseSummary[];
  steady_state?: boolean;
  route?: string;
  blq?: { n_below_loq: number; rule: string };
}

export interface DoseSummary {
  dose: number;
  n: number;
  Cmax_geomean: number;
  Cmax_geocv_pct: number | null;
  AUC_last_geomean: number;
  AUC_last_geocv_pct: number | null;
  AUC_inf_geomean: number;
  AUC_inf_geocv_pct: number | null;
  CL_F_geomean: number;
  CL_F_geocv_pct: number | null;
  Vz_F_geomean: number;
  Vz_F_geocv_pct: number | null;
  t_half_median: number;
}

export interface QcIssue {
  severity: 'HIGH' | 'MEDIUM' | 'LOW';
  issue: string;
}

export interface QcCheck {
  check: string;
  status: 'PASS' | 'WARN' | 'FAIL';
  detail: string;
}

export interface AuditEntry {
  index: number;
  timestamp: string;
  agent: string;
  tool: string;
  action: string;
  inputs_hash: string;
  outputs_hash: string;
  prev_hash: string;
  entry_hash: string;
  actor?: string;
  reason?: string;
}

export interface ChatResponse {
  agent: string;
  messages: AgentMessage[];
  state: PharmState;
  status: string;
}

export interface WorkflowResponse {
  status: 'complete' | 'awaiting_review' | 'error';
  state: PharmState;
  messages?: AgentMessage[];
  pending_review?: PendingReview;
  audit_ok: boolean;
}

export interface PendingReview {
  step: number;
  step_name: string;
  prompt: string;
  state_snapshot: PharmState;
}

export interface AgentMessage {
  role: 'user' | 'assistant';
  content: string | ContentBlock[];
}

export interface ContentBlock {
  type: 'text' | 'tool_use' | 'tool_result';
  text?: string;
  name?: string;
  input?: Record<string, unknown>;
}

export type WorkflowStatus = 'idle' | 'running' | 'awaiting_review' | 'complete' | 'error';

// ── Flexplot (exploratory visualization) ──────────────────────────────────
export interface VariableMeta {
  name: string;
  dtype: string;
  role: string;
  type: 'continuous' | 'categorical';
  is_numeric: boolean;
  n_unique: number | null;
  min: number | null;
  max: number | null;
}
export interface VariablesResponse {
  variables: VariableMeta[];
  detected_roles: Record<string, string>;
}

export type FlexplotFitKind = 'loess' | 'linear' | 'none';
export type FlexplotGeom = 'points' | 'line' | 'smooth' | 'density';
export type FlexplotCenter = 'median_iqr' | 'mean_se' | 'mean_sd';

export interface FlexplotSpec {
  y: string;
  x: string | null;
  color_by: string | null;
  panel_by: string | null;
  fit: FlexplotFitKind;
  geom: FlexplotGeom;
  center: FlexplotCenter;
  ghost: boolean;
  log_y: boolean;
  jitter: number;
  n_bins: number;
  ci: number;
}

export interface FlexplotPoints { x: number[]; y: number[]; }
export interface FlexplotFit { x: number[]; y: number[]; lo: number[]; hi: number[]; }
export interface FlexplotCrossbar {
  group_x: string; x_index: number;
  center: number | null; lo: number | null; hi: number | null; n: number;
}
export interface FlexplotBins { edges: number[]; counts: number[]; }
export interface FlexplotDensity { x: number[]; y: number[]; }
export interface FlexplotCell {
  panel: string;
  group: string;
  points: FlexplotPoints;
  fit: FlexplotFit | null;
  crossbars: FlexplotCrossbar[];
  bins: FlexplotBins | null;
  density: FlexplotDensity | null;
  n: number;
}
export interface FlexplotLegendEntry { id: string; label: string; color_index: number; }
export interface FlexplotPanelMeta { id: string; strip: string; bin_range: [number, number] | null; }
export interface FlexplotGhostLine { group: string; color_index: number; x: number[]; y: number[]; }
export interface FlexplotSummary {
  n: number;
  x_mean: number | null; y_mean: number | null;
  n_groups: number; n_panels: number;
}
export interface FlexplotData {
  kind: 'scatter' | 'dotplot' | 'histogram' | 'density';
  echo: Record<string, unknown>;
  var_types: {
    x: 'continuous' | 'categorical' | null;
    y: 'continuous' | 'categorical';
    color_by: 'categorical' | null;
    panel_by: 'categorical' | 'continuous_binned' | null;
  };
  x_label: string;
  y_label: string;
  x_range: [number, number];
  y_range: [number, number];
  log_scale: boolean;
  x_categories: string[] | null;
  groups: string[];
  legend: FlexplotLegendEntry[];
  panels: FlexplotPanelMeta[];
  cells: FlexplotCell[];
  ghost_line: FlexplotGhostLine[] | null;
  summary: FlexplotSummary;
}
