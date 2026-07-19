// Run the SAME parity harness under Pyodide/WASM (Node host).
//
//   npm install          # once — pulls the `pyodide` package
//   node run_pyodide.mjs [--quick]
//
// Loads Pyodide + numpy/scipy/pandas, mounts backend/app + harness.py + the
// Theophylline CSV into the WASM virtual filesystem, runs harness.run(), and
// writes results_pyodide.json. Then run compare.py to diff vs CPython.
//
// Note the scipy/pandas versions it prints — Pyodide ships OLDER pins than the
// backend (the numerical-parity risk in docs/WASM_BROWSER_NATIVE_SPEC.md §6).
import { loadPyodide } from "pyodide";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const BACKEND = path.resolve(HERE, "..", "backend");
const APP = path.join(BACKEND, "app");
const CSV = path.join(BACKEND, "sample_data", "theoph_pk.csv");
const QUICK = process.argv.includes("--quick");

// Collect every .py under backend/app (skip __pycache__) as [relpath, source].
function collectPy(dir, base, out) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.name === "__pycache__") continue;
    const full = path.join(dir, entry.name);
    const rel = path.posix.join(base, entry.name);
    if (entry.isDirectory()) collectPy(full, rel, out);
    else if (entry.name.endsWith(".py")) out.push([rel, fs.readFileSync(full, "utf8")]);
  }
}

async function main() {
  const t0 = Date.now();
  console.log("[pyodide] loading runtime + numpy/scipy/pandas (first run downloads ~30-40 MB)…");
  const py = await loadPyodide();
  // pandas/scipy/numpy for compute; pydantic for app.core.pharmstate (PharmState).
  await py.loadPackage(["numpy", "scipy", "pandas", "pydantic"]);

  // Mount app package + harness + data onto sys.path inside the WASM FS.
  const files = [];
  collectPy(APP, "app", files);
  for (const [rel, source] of files) {
    const dest = `/session/${rel}`;
    py.FS.mkdirTree(path.posix.dirname(dest));
    py.FS.writeFile(dest, source);
  }
  py.FS.writeFile("/session/harness.py", fs.readFileSync(path.join(HERE, "harness.py"), "utf8"));
  py.FS.writeFile("/session/theoph_pk.csv", fs.readFileSync(CSV, "utf8"));

  console.log(`[pyodide] mounted ${files.length} app modules; running harness (${QUICK ? "quick" : "full"} mode)…`);
  const resultJson = await py.runPythonAsync(`
import sys, json
sys.path.insert(0, "/session")
import harness
json.dumps(harness.run("/session/theoph_pk.csv", quick=${QUICK ? "True" : "False"}))
`);

  const outPath = path.join(HERE, "results_pyodide.json");
  const result = JSON.parse(resultJson);
  fs.writeFileSync(outPath, JSON.stringify(result, null, 2));
  const v = result.versions;
  console.log(`[pyodide] wrote ${outPath}`);
  console.log(`[pyodide] numpy ${v.numpy}  scipy ${v.scipy}  pandas ${v.pandas}  (${result.mode} mode)`);
  console.log(`[pyodide] FOCE-I CL=${result.focei.theta.CL.toFixed(4)}  SAEM CL=${result.saem.theta.CL.toFixed(4)}  ` +
              `converged=${result.focei.converged}/${result.saem.converged}  in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
}

main().catch((e) => { console.error(e); process.exit(1); });
