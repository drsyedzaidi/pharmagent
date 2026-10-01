/** Display names for backend agent ids. Unknown ids fall back to the raw id. */
export const AGENT_LABEL: Record<string, string> = {
  supervisor: 'Supervisor',
  data_manager: 'Data manager',
  nca: 'NCA agent',
  qc: 'QC agent',
  report: 'Report agent',
  modeler: 'Modeler',
  simulator: 'Simulator',
  reviewer: 'Reviewer',
  clinpharm: 'Clinical pharmacology',
  statistician: 'Statistician',
  be: 'Bioequivalence agent',
  dose_prop: 'Dose-proportionality agent',
  compartmental: 'Compartmental agent',
  poppk: 'Population PK agent',
  system: 'System',
};

export function agentLabel(id?: string): string {
  if (!id) return '';
  return AGENT_LABEL[id] ?? id;
}
