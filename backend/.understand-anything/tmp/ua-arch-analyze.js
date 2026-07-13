#!/usr/bin/env node
'use strict';

const fs = require('fs');

function fail(msg) { process.stderr.write(String(msg) + '\n'); process.exit(1); }

const inPath = process.argv[2];
const outPath = process.argv[3];
if (!inPath || !outPath) fail('usage: node ua-arch-analyze.js <input.json> <output.json>');

let data;
try { data = JSON.parse(fs.readFileSync(inPath, 'utf8')); }
catch (e) { fail('cannot read/parse input: ' + e.message); }

const fileNodes = Array.isArray(data.fileNodes) ? data.fileNodes : [];
const importEdges = Array.isArray(data.importEdges) ? data.importEdges : [];
const allEdges = Array.isArray(data.allEdges) ? data.allEdges : [];
if (fileNodes.length === 0) fail('no fileNodes in input');

const idToNode = new Map();
for (const n of fileNodes) idToNode.set(n.id, n);
const ids = fileNodes.map(n => n.id);

function fp(n) { return (n.filePath || n.name || '').replace(/^[.][/]/, ''); }

// ---- Common prefix across directory portions ----
const dirSegsList = fileNodes.map(n => {
  const p = fp(n);
  const parts = p.split('/');
  return parts.slice(0, -1); // directory segments only
});
// common prefix of directory segments (only meaningful if every file is nested)
let commonPrefix = [];
if (dirSegsList.length > 0) {
  const minLen = Math.min(...dirSegsList.map(s => s.length));
  for (let i = 0; i < minLen; i++) {
    const seg = dirSegsList[0][i];
    if (dirSegsList.every(s => s[i] === seg)) commonPrefix.push(seg); else break;
  }
  // do not strip a prefix that swallows ALL directory depth for some file
  // (keep as-is; grouping below handles root files)
}

function groupKey(n) {
  const p = fp(n);
  const parts = p.split('/');
  const dirs = parts.slice(0, -1);
  if (dirs.length === 0) return '(root)';
  let rel = dirs;
  if (commonPrefix.length > 0 && commonPrefix.length < dirs.length) {
    // strip common prefix only when file is deeper than prefix
    let matches = true;
    for (let i = 0; i < commonPrefix.length; i++) if (dirs[i] !== commonPrefix[i]) { matches = false; break; }
    if (matches) rel = dirs.slice(commonPrefix.length);
  }
  return rel[0] || '(root)';
}

// ---- A. Directory grouping ----
const directoryGroups = {};
for (const n of fileNodes) {
  const k = groupKey(n);
  (directoryGroups[k] = directoryGroups[k] || []).push(n.id);
}

// ---- B. Node type grouping ----
const nodeTypeGroups = {};
for (const n of fileNodes) {
  const t = n.type || 'file';
  (nodeTypeGroups[t] = nodeTypeGroups[t] || []).push(n.id);
}

// ---- group lookup per id ----
const idToGroup = new Map();
for (const [g, arr] of Object.entries(directoryGroups)) for (const id of arr) idToGroup.set(id, g);

// ---- C. Import adjacency / fan-in / fan-out ----
const fanOut = {}; const fanIn = {};
for (const id of ids) { fanOut[id] = 0; fanIn[id] = 0; }
for (const e of importEdges) {
  if (fanOut[e.source] === undefined || fanIn[e.target] === undefined) continue;
  fanOut[e.source]++; fanIn[e.target]++;
}

// ---- D. Cross-category dependency analysis (allEdges, cross node-type) ----
const crossCatMap = new Map();
for (const e of allEdges) {
  const s = idToNode.get(e.source), t = idToNode.get(e.target);
  if (!s || !t) continue;
  const st = s.type || 'file', tt = t.type || 'file';
  if (st === tt) continue; // cross-category only
  const key = st + '|' + tt + '|' + (e.type || 'rel');
  crossCatMap.set(key, (crossCatMap.get(key) || 0) + 1);
}
const crossCategoryEdges = [];
for (const [k, count] of crossCatMap.entries()) {
  const [fromType, toType, edgeType] = k.split('|');
  crossCategoryEdges.push({ fromType, toType, edgeType, count });
}
crossCategoryEdges.sort((a, b) => b.count - a.count);

// ---- E. Inter-group import frequency (use importEdges) ----
const interMap = new Map();
for (const e of importEdges) {
  const gs = idToGroup.get(e.source), gt = idToGroup.get(e.target);
  if (gs === undefined || gt === undefined || gs === gt) continue;
  const key = gs + '||' + gt;
  interMap.set(key, (interMap.get(key) || 0) + 1);
}
const interGroupImports = [];
for (const [k, count] of interMap.entries()) {
  const [from, to] = k.split('||');
  interGroupImports.push({ from, to, count });
}
interGroupImports.sort((a, b) => b.count - a.count);

// ---- F. Intra-group import density ----
const intraGroupDensity = {};
for (const g of Object.keys(directoryGroups)) intraGroupDensity[g] = { internalEdges: 0, totalEdges: 0, density: 0 };
for (const e of importEdges) {
  const gs = idToGroup.get(e.source), gt = idToGroup.get(e.target);
  if (gs !== undefined) intraGroupDensity[gs].totalEdges++;
  if (gt !== undefined && gt !== gs) intraGroupDensity[gt].totalEdges++;
  if (gs !== undefined && gs === gt) intraGroupDensity[gs].internalEdges++;
}
for (const g of Object.keys(intraGroupDensity)) {
  const d = intraGroupDensity[g];
  d.density = d.totalEdges > 0 ? +(d.internalEdges / d.totalEdges).toFixed(3) : 0;
}

// ---- G. Directory + file pattern matching ----
const DIR_PATTERNS = [
  [['routes','api','controllers','endpoints','handlers'], 'api'],
  [['services','core','lib','domain','logic'], 'service'],
  [['models','db','data','persistence','repository','entities'], 'data'],
  [['components','views','pages','ui','layouts','screens'], 'ui'],
  [['middleware','plugins','interceptors','guards'], 'middleware'],
  [['utils','helpers','common','shared','tools'], 'utility'],
  [['config','constants','env','settings'], 'config'],
  [['__tests__','test','tests','spec','specs'], 'test'],
  [['types','interfaces','schemas','contracts','dtos'], 'types'],
  [['hooks'], 'hooks'],
  [['store','state','reducers','actions','slices'], 'state'],
  [['assets','static','public'], 'assets'],
  [['migrations'], 'data'],
  [['management','commands'], 'config'],
  [['templatetags'], 'utility'],
  [['signals'], 'service'],
  [['serializers'], 'api'],
  [['cmd'], 'entry'],
  [['internal'], 'service'],
  [['pkg'], 'utility'],
  [['agents'], 'agents'],
  [['compute'], 'service'],
  [['docs','documentation','wiki'], 'documentation'],
  [['deploy','deployment','infra','infrastructure'], 'infrastructure'],
  [['.github','.gitlab','.circleci'], 'ci-cd'],
  [['k8s','kubernetes','helm','charts'], 'infrastructure'],
  [['terraform','tf'], 'infrastructure'],
  [['docker'], 'infrastructure'],
  [['sql','database'], 'data'],
];
const dirPatternMap = new Map();
for (const [names, label] of DIR_PATTERNS) for (const nm of names) dirPatternMap.set(nm, label);

function fileLevelPattern(n) {
  const p = fp(n);
  const base = p.split('/').pop();
  const t = n.type || 'file';
  if (/(\.test\.|\.spec\.)/.test(base) || /^test_.*\.py$/.test(base) || /_test\.go$/.test(base) || /Test\.java$/.test(base) || /_spec\.rb$/.test(base) || /Test\.php$/.test(base) || /Tests\.cs$/.test(base)) return 'test';
  if (/\.d\.ts$/.test(base)) return 'types';
  if (base === 'manage.py') return 'entry';
  if (base === 'wsgi.py' || base === 'asgi.py') return 'config';
  if (base === 'main.py' && /(^|\/)app\//.test(p) === false && p.split('/').length <= 2) return 'entry';
  if (/^(Cargo\.toml|go\.mod|Gemfile|pom\.xml|build\.gradle|composer\.json)$/.test(base)) return 'config';
  if (base === 'Dockerfile' || /^docker-compose\./.test(base) || base === '.dockerignore') return 'infrastructure';
  if (/\.(tf|tfvars)$/.test(base)) return 'infrastructure';
  if (base === '.gitlab-ci.yml' || base === 'Jenkinsfile') return 'ci-cd';
  if (/^\.github\/workflows\//.test(p)) return 'ci-cd';
  if (/\.sql$/.test(base)) return 'data';
  if (/\.(graphql|gql|proto)$/.test(base)) return 'types';
  if (/\.(md|rst)$/.test(base)) return 'documentation';
  if (base === 'Makefile') return 'infrastructure';
  if (t === 'document') return 'documentation';
  if (t === 'config') return 'config';
  if (t === 'service') return 'infrastructure';
  if (t === 'pipeline') return 'ci-cd';
  if (t === 'table' || t === 'schema') return 'data';
  return null;
}

const patternMatches = {};
for (const g of Object.keys(directoryGroups)) {
  if (dirPatternMap.has(g)) patternMatches[g] = dirPatternMap.get(g);
}
// file-level pattern hints aggregated per group (for ambiguous groups)
const fileLevelPatterns = {};
for (const n of fileNodes) {
  const fl = fileLevelPattern(n);
  if (fl) fileLevelPatterns[n.id] = fl;
}

// ---- H. Deployment topology ----
const infraFiles = [];
let hasDockerfile=false, hasCompose=false, hasK8s=false, hasTerraform=false, hasCI=false;
for (const n of fileNodes) {
  const p = fp(n); const base = p.split('/').pop();
  if (base === 'Dockerfile') { hasDockerfile = true; infraFiles.push(p); }
  if (/^docker-compose\./.test(base)) { hasCompose = true; infraFiles.push(p); }
  if (base === '.dockerignore') infraFiles.push(p);
  if (/\.(ya?ml)$/.test(base) && /(k8s|kube|deployment|service|ingress)/i.test(p)) { hasK8s = true; infraFiles.push(p); }
  if (/\.(tf|tfvars)$/.test(base)) { hasTerraform = true; infraFiles.push(p); }
  if (/^\.github\/workflows\//.test(p) || base === '.gitlab-ci.yml' || base === 'Jenkinsfile') { hasCI = true; infraFiles.push(p); }
  if ((n.type||'') === 'pipeline') { hasCI = true; if (!infraFiles.includes(p)) infraFiles.push(p); }
}
const deploymentTopology = { hasDockerfile, hasCompose, hasK8s, hasTerraform, hasCI, infraFiles };

// ---- I. Data pipeline ----
const schemaFiles=[], migrationFiles=[], dataModelFiles=[], apiHandlerFiles=[];
for (const n of fileNodes) {
  const p = fp(n); const base = p.split('/').pop();
  if (/\.(sql|graphql|gql|proto|prisma)$/.test(base)) schemaFiles.push(p);
  if (/(^|\/)migrations(\/|$)/.test(p)) migrationFiles.push(p);
  const g = idToGroup.get(n.id);
  if (g && dirPatternMap.get(g) === 'data') dataModelFiles.push(p);
  if (g && dirPatternMap.get(g) === 'api') apiHandlerFiles.push(p);
}
const dataPipeline = { schemaFiles, migrationFiles, dataModelFiles, apiHandlerFiles };

// ---- J. Documentation coverage ----
const docNodes = fileNodes.filter(n => (n.type||'')==='document' || /\.(md|rst)$/.test(fp(n)));
const groupHasDoc = new Set();
for (const n of docNodes) groupHasDoc.add(idToGroup.get(n.id));
const totalGroups = Object.keys(directoryGroups).length;
const groupsWithDocs = [...groupHasDoc].filter(Boolean).length;
const undocumentedGroups = Object.keys(directoryGroups).filter(g => !groupHasDoc.has(g));
const docCoverage = {
  groupsWithDocs, totalGroups,
  coverageRatio: totalGroups ? +(groupsWithDocs/totalGroups).toFixed(2) : 0,
  undocumentedGroups
};

// ---- K. Dependency direction ----
const pairNet = new Map();
for (const ig of interGroupImports) {
  const a = ig.from, b = ig.to;
  const key = [a,b].sort().join('||');
  const cur = pairNet.get(key) || { a: a<b?a:b, b: a<b?b:a, ab:0, ba:0 };
  if (a === cur.a) cur.ab += ig.count; else cur.ba += ig.count;
  pairNet.set(key, cur);
}
const dependencyDirection = [];
for (const v of pairNet.values()) {
  if (v.ab > v.ba) dependencyDirection.push({ dependent: v.a, dependsOn: v.b });
  else if (v.ba > v.ab) dependencyDirection.push({ dependent: v.b, dependsOn: v.a });
}

// ---- file stats ----
const filesPerGroup = {};
for (const [g, arr] of Object.entries(directoryGroups)) filesPerGroup[g] = arr.length;
const nodeTypeCounts = {};
for (const [t, arr] of Object.entries(nodeTypeGroups)) nodeTypeCounts[t] = arr.length;

const result = {
  scriptCompleted: true,
  commonPrefix: commonPrefix.join('/'),
  directoryGroups,
  nodeTypeGroups,
  crossCategoryEdges,
  interGroupImports,
  intraGroupDensity,
  patternMatches,
  fileLevelPatterns,
  deploymentTopology,
  dataPipeline,
  docCoverage,
  dependencyDirection,
  fileStats: { totalFileNodes: fileNodes.length, filesPerGroup, nodeTypeCounts },
  fileFanIn: fanIn,
  fileFanOut: fanOut
};

try { fs.writeFileSync(outPath, JSON.stringify(result, null, 2)); }
catch (e) { fail('cannot write output: ' + e.message); }
process.exit(0);
