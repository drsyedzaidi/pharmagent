#!/usr/bin/env node
"use strict";

// Tour topology analyzer for PharmAgent backend.
// Usage: node ua-tour-analyze.js <input.json> <output.json>

const fs = require("fs");

function main() {
  const inPath = process.argv[2];
  const outPath = process.argv[3];
  if (!inPath || !outPath) {
    console.error("Usage: node ua-tour-analyze.js <input.json> <output.json>");
    process.exit(1);
  }

  const raw = JSON.parse(fs.readFileSync(inPath, "utf8"));
  const nodes = raw.nodes || [];
  const edges = raw.edges || [];
  const layers = raw.layers || [];

  const idSet = new Set(nodes.map((n) => n.id));
  const byId = new Map(nodes.map((n) => [n.id, n]));

  // --- Node summary index (all node types) ---
  const nodeSummaryIndex = {};
  for (const n of nodes) {
    nodeSummaryIndex[n.id] = {
      name: n.name || "",
      type: n.type || "",
      summary: n.summary || "",
    };
  }

  // --- Fan-in / Fan-out over ALL edge types (drop self loops, dedupe pairs per type) ---
  const fanIn = new Map();
  const fanOut = new Map();
  for (const id of idSet) {
    fanIn.set(id, 0);
    fanOut.set(id, 0);
  }
  const seenEdge = new Set();
  for (const e of edges) {
    if (!e || !idSet.has(e.source) || !idSet.has(e.target)) continue;
    if (e.source === e.target) continue;
    const key = e.source + "|" + e.target + "|" + (e.type || "");
    if (seenEdge.has(key)) continue;
    seenEdge.add(key);
    fanOut.set(e.source, fanOut.get(e.source) + 1);
    fanIn.set(e.target, fanIn.get(e.target) + 1);
  }

  const fanInRanking = [...fanIn.entries()]
    .map(([id, c]) => ({ id, fanIn: c, name: (byId.get(id) || {}).name || "" }))
    .sort((a, b) => b.fanIn - a.fanIn || a.id.localeCompare(b.id))
    .slice(0, 20);

  const fanOutRanking = [...fanOut.entries()]
    .map(([id, c]) => ({ id, fanOut: c, name: (byId.get(id) || {}).name || "" }))
    .sort((a, b) => b.fanOut - a.fanOut || a.id.localeCompare(b.id))
    .slice(0, 20);

  // --- Entry point candidates ---
  const codeEntryNames = new Set([
    "index.ts", "index.js", "main.ts", "main.js", "app.ts", "app.js",
    "server.ts", "server.js", "mod.rs", "main.go", "main.py", "main.rs",
    "manage.py", "app.py", "wsgi.py", "asgi.py", "run.py", "__main__.py",
    "Application.java", "Main.java", "Program.cs", "config.ru", "index.php",
    "App.swift", "Application.kt", "main.cpp", "main.c",
  ]);

  // thresholds for fan-out top 10% and fan-in bottom 25% among code files
  const codeFiles = nodes.filter((n) => n.type === "file");
  const foSorted = codeFiles
    .map((n) => fanOut.get(n.id) || 0)
    .sort((a, b) => b - a);
  const fiSorted = codeFiles
    .map((n) => fanIn.get(n.id) || 0)
    .sort((a, b) => a - b);
  const foTop10Idx = Math.max(0, Math.floor(foSorted.length * 0.1) - 1);
  const foTop10Threshold = foSorted.length ? foSorted[foTop10Idx] : 0;
  const fiBottom25Idx = Math.max(0, Math.floor(fiSorted.length * 0.25) - 1);
  const fiBottom25Threshold = fiSorted.length ? fiSorted[fiBottom25Idx] : 0;

  function depthFromRoot(fp) {
    if (!fp) return 99;
    return fp.split("/").filter(Boolean).length - 1; // file at root => 0
  }

  const entryScores = [];
  for (const n of nodes) {
    let score = 0;
    const fp = n.filePath || "";
    const name = n.name || "";
    if (n.type === "file") {
      if (codeEntryNames.has(name)) score += 3;
      if (depthFromRoot(fp) <= 1) score += 1;
      if ((fanOut.get(n.id) || 0) >= foTop10Threshold && foTop10Threshold > 0) score += 1;
      if ((fanIn.get(n.id) || 0) <= fiBottom25Threshold) score += 1;
    } else if (n.type === "document") {
      const base = name.toLowerCase();
      const atRoot = depthFromRoot(fp) === 0;
      if (base === "readme.md" && atRoot) score += 5;
      else if (base.endsWith(".md") && atRoot) score += 2;
    }
    if (score > 0) {
      entryScores.push({
        id: n.id,
        score,
        name,
        summary: n.summary || "",
        type: n.type,
      });
    }
  }
  entryScores.sort((a, b) => b.score - a.score || a.id.localeCompare(b.id));
  const entryPointCandidates = entryScores.slice(0, 5);

  // --- BFS from top CODE entry point following imports + calls (forward) ---
  const codeEntry = entryScores.find((c) => {
    const t = (byId.get(c.id) || {}).type;
    return t === "file";
  });
  const startNode = codeEntry ? codeEntry.id : (codeFiles[0] ? codeFiles[0].id : null);

  const adj = new Map();
  for (const id of idSet) adj.set(id, []);
  for (const e of edges) {
    if (!idSet.has(e.source) || !idSet.has(e.target)) continue;
    if (e.type === "imports" || e.type === "calls") {
      adj.get(e.source).push(e.target);
    }
  }

  const order = [];
  const depthMap = {};
  if (startNode) {
    const q = [startNode];
    depthMap[startNode] = 0;
    const visited = new Set([startNode]);
    while (q.length) {
      const cur = q.shift();
      order.push(cur);
      const nbrs = [...new Set(adj.get(cur) || [])].sort();
      for (const nb of nbrs) {
        if (!visited.has(nb)) {
          visited.add(nb);
          depthMap[nb] = depthMap[cur] + 1;
          q.push(nb);
        }
      }
    }
  }
  const byDepth = {};
  for (const [id, d] of Object.entries(depthMap)) {
    const k = String(d);
    if (!byDepth[k]) byDepth[k] = [];
    byDepth[k].push(id);
  }
  for (const k of Object.keys(byDepth)) byDepth[k].sort();

  // --- Non-code file inventory ---
  const documentation = [];
  const infrastructure = [];
  const data = [];
  const config = [];
  for (const n of nodes) {
    const entry = { id: n.id, name: n.name || "", summary: n.summary || "" };
    if (n.type === "document") documentation.push(entry);
    else if (["service", "pipeline", "resource"].includes(n.type)) infrastructure.push(entry);
    else if (["table", "schema", "endpoint"].includes(n.type)) data.push(entry);
    else if (n.type === "config") config.push(entry);
  }

  // --- Tightly coupled clusters (bidirectional imports/calls, then expand) ---
  const relTypes = new Set(["imports", "calls", "depends_on"]);
  const dir = new Map(); // "a|b" -> count of directed rel edges
  for (const e of edges) {
    if (!idSet.has(e.source) || !idSet.has(e.target)) continue;
    if (e.source === e.target) continue;
    if (!relTypes.has(e.type)) continue;
    const k = e.source + "|" + e.target;
    dir.set(k, (dir.get(k) || 0) + 1);
  }
  // undirected edge weight + adjacency among relTypes
  const undW = new Map(); // "min|max" -> weight
  const relAdj = new Map();
  for (const id of idSet) relAdj.set(id, new Set());
  for (const [k, c] of dir.entries()) {
    const [a, b] = k.split("|");
    relAdj.get(a).add(b);
    relAdj.get(b).add(a);
    const uk = a < b ? a + "|" + b : b + "|" + a;
    undW.set(uk, (undW.get(uk) || 0) + c);
  }
  // seeds = bidirectional pairs (A->B and B->A both present)
  const seeds = [];
  for (const [uk, w] of undW.entries()) {
    const [a, b] = uk.split("|");
    if (dir.has(a + "|" + b) && dir.has(b + "|" + a)) {
      seeds.push([a, b]);
    }
  }
  const clusters = [];
  const usedSig = new Set();
  function clusterEdgeCount(members) {
    let cnt = 0;
    const set = new Set(members);
    for (const m of members) {
      for (const nb of relAdj.get(m) || []) {
        if (set.has(nb)) cnt += 1; // counts each directed adjacency once
      }
    }
    return cnt;
  }
  for (const [a, b] of seeds) {
    const members = new Set([a, b]);
    // expand: add nodes connected to >=2 existing members, cap 5
    let changed = true;
    while (changed && members.size < 5) {
      changed = false;
      const cand = new Map();
      for (const m of members) {
        for (const nb of relAdj.get(m) || []) {
          if (!members.has(nb)) cand.set(nb, (cand.get(nb) || 0) + 1);
        }
      }
      let best = null;
      let bestC = 1;
      for (const [nb, c] of cand.entries()) {
        if (c >= 2 && c > bestC) {
          best = nb;
          bestC = c;
        }
      }
      if (best) {
        members.add(best);
        changed = true;
      }
    }
    const arr = [...members].sort();
    const sig = arr.join(",");
    if (usedSig.has(sig)) continue;
    usedSig.add(sig);
    clusters.push({ nodes: arr, edgeCount: clusterEdgeCount(arr) });
  }
  clusters.sort((a, b) => b.edgeCount - a.edgeCount || b.nodes.length - a.nodes.length);
  const topClusters = clusters.slice(0, 10);

  // --- Layers ---
  const layerOut = {
    count: layers.length,
    list: layers.map((l) => ({
      id: l.id,
      name: l.name,
      description: l.description || "",
    })),
  };

  const result = {
    scriptCompleted: true,
    entryPointCandidates,
    fanInRanking,
    fanOutRanking,
    bfsTraversal: {
      startNode,
      order,
      depthMap,
      byDepth,
    },
    nonCodeFiles: { documentation, infrastructure, data, config },
    clusters: topClusters,
    layers: layerOut,
    nodeSummaryIndex,
    totalNodes: nodes.length,
    totalEdges: edges.length,
  };

  fs.writeFileSync(outPath, JSON.stringify(result, null, 2));
  console.error(
    "OK nodes=" + nodes.length + " edges=" + edges.length +
    " start=" + startNode + " bfsReached=" + order.length +
    " clusters=" + topClusters.length
  );
  process.exit(0);
}

try {
  main();
} catch (err) {
  console.error("FATAL: " + (err && err.stack ? err.stack : err));
  process.exit(1);
}
