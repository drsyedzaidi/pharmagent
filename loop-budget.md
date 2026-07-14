# Loop Budget — PharmAgent

> The `loop-budget` skill reads this at the **start and end** of every loop run.

## Daily limits

| Loop | Max runs/day | Max tokens/day | Max sub-agent spawns/run |
|------|--------------|----------------|--------------------------|
| CI Sweeper | 24 | 400k | 2 (maker + verifier) |
| Security-Finding Triage | 2 | 150k | 1 |
| Dependency Sweeper | 1 | 100k | 1 |

## Admission control (compute jobs)

PharmAgent's expensive path is NLME (FOCE/SAEM), SCM, and engine-comparison
(`backend/app/core/jobs.py`). These are **not** loop-cheap. A loop must **never**
submit a real fit; if a proposed fix requires running one, it escalates to a human.
> This preserves the post-audit invariant from `c9d2658`: no loop may enqueue
> long-running jobs without a documented cap and human trigger.

## On budget exceed
1. Switch to **report-only** at 80% of the daily token cap.
2. Append a `"outcome":"budget-exceeded"` entry to `loop-run-log.md`.
3. Stop and escalate to a human; do not start a new item.

## Kill switch
- Set `loop-pause-all` on the first line of `STATE.md`.
- Any running loop must check for it at start and **exit immediately** if present.
- Resume only after the flag is cleared in `STATE.md` by a human.

## Estimate spend
```bash
npx @cobusgreyling/loop-cost --pattern ci-sweeper --level L1
```
