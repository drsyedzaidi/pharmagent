# PMxbench scenario_00 — public payload (test fixture)

Three files copied byte-for-byte from the ISoP AI/ML SIG PMxbench benchmark:

- Repository: https://github.com/AIML-SIG/Agentic-workflows
- Commit: `e7de2630c5906d74e31681baa53bbd32ddb2cf33` (main, 2026-10-06)
- Folder: `tools/pmxbench/scenario_00/`
- Licence: MIT. `tools/pmxbench/LICENSE` (Copyright (c) 2026 Ari Pritchard-Bell) is
  copied here as `LICENSE`; the repository root is also MIT (Copyright (c) 2026
  AI/ML Special Interest Group).

| File | Upstream git blob SHA |
|---|---|
| `data.csv` | `b2f9b2371eec482d392761421cbbec2bfa780572` |
| `sap.md` | `ca3edd5b4b7eb32cfc058ec4f57b6dccc5cf6f8c` |
| `submission.template.yaml` | `bd7a1b677893f49f53f71b62529e6afaab7d26d8` |

`MANIFEST.json` pins the blob SHAs and SHA-256 digests; `tests/test_pmxbench.py`
fails if a file drifts.

Fetched with:

```bash
gh api -H "Accept: application/vnd.github.raw" \
  "repos/AIML-SIG/Agentic-workflows/contents/tools/pmxbench/scenario_00/<file>?ref=e7de2630c5906d74e31681baa53bbd32ddb2cf33"
```

## No answer key, ever

Only the three public payload files are here. The upstream answer key and the
data generator in the maintainers' template folder are **never** fetched, copied,
read into code or referenced by this repository. A test asserts that no file named
`truth.*`, `generate.R` or `score.py` exists anywhere in the repository. Scoring
happens outside PharmAgent; see `docs/pmxbench.md`.
