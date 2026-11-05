# M4-A portfolio demo package — local Tier A

**Status:** local foundation verified, 2026-11-05; CI and publication remain separate gates. **Publication:** not authorized. **Source baseline:** `ba8d098` plus this branch's demo polish. The safe footage is a generated one-second solid-color video. Its `demo_fast` artifacts are synthetic placeholders, not YOLO results or match statistics.

## Tier readiness

| Tier | What could be shown | Gate |
|---|---|---|
| A — deterministic, generated fixture | Local dashboard, synthetic run, citations, scripted tool trace, unsupported-question abstention | Ready for private recording rehearsal; owner still controls publication |
| B — shareable real CV | Real inference on footage with documented public rights | **BLOCKED:** no shareable real footage or rights record. Run `87da5c1b` used 2018 World Cup broadcast footage and remains private |
| C — real model agent | Model-selected tools and claims, evaluated against accepted rules and reviewed labels | **UNVERIFIED LIVE / BLOCKED:** no measured Anthropic agent quality; v3 independent review is 0/142 |

## 60-second Tier-A storyboard — recordable on generated content

| Time | Shot and exact action | On-screen or voice caption | Evidence boundary |
|---|---|---|---|
| 0–7 s | Title card, then local dashboard | “FootballAI: evidence-grounded video analysis, demonstrated with a synthetic fixture.” | Show `LOCAL DEMO · SYNTHETIC FIXTURE` throughout the UI footage |
| 7–17 s | Analysis runs → generated run detail; hold on profile and status | “A queue and worker produce an immutable run with integrity-checked artifacts. This fixture uses `demo_fast`.” | Do not show this as YOLO inference |
| 17–29 s | Copilot, Grounded Explanation; ask “What is the status of this run?” | “The server answers from registered evidence and cites its source fields.” | Hold long enough to read `Synthetic output` and a citation |
| 29–42 s | Switch to Read-Only Agent; ask “What do the tracking diagnostics tell us?”; expand trace | “A fixed script chooses an approved read-only tool. The trace shows the tool that actually ran.” | Show `OFFLINE SCRIPTED AGENT` and `get_tracking_diagnostics`; do not call it an LLM |
| 42–51 s | Ask “How many successful passes did player 7 make?”; expand trace | “Pass recognition is outside this system, so it abstains before any tool call.” | Show no citations and “No tool ran” |
| 51–60 s | Current architecture SVG and repository title | “FastAPI, React, provider-neutral storage and queue ports, and server-side grounding. Local demonstration.” | Diagram labels distinguish local, historical Azure, unverified live and planned |

No real match video, real-run ID, private artifact, API key, cloud portal or personal account screen appears in Tier A. The scripted trace proves orchestration plumbing and local safety checks, not model judgement.

## Optional 85-second Tier-C storyboard — FUTURE / UNVERIFIED

**Do not record or publish this as a current demonstration.** It requires human-reviewed labels, an approved live evaluation that passes the accepted rule, and shareable real footage. If those gates fail, keep Tier A.

| Time | Future shot | Required evidence before caption is allowed |
|---|---|---|
| 0–9 s | Title and rights-cleared source | Documented licence and consent for public use |
| 9–23 s | Real run with `verified_real` execution label | A new run on that source; avoid claiming accuracy |
| 23–40 s | Live model mode, relevant tool trace | Actual paid evaluation and route trace, with model mode visible |
| 40–54 s | Citation drill-down | Citation and artifact integrity check on that run |
| 54–65 s | Unsupported passes question abstains | Live result on the same tested configuration |
| 65–76 s | Evaluation report beside deterministic baseline | Accepted decision rule passed on human-reviewed labels; include failures |
| 76–85 s | Current architecture and repository | Fresh diagram, cloud status rechecked before any cloud claim |

## Exact local recording rehearsal

From a clean checkout on macOS or Linux, install Python 3.13, Node/npm and FFmpeg. Then from the repository root:

```bash
python3.13 -m venv .venv-test
.venv-test/bin/python -m pip install -r requirements-test.txt
cd v2/dashboard && npm ci --no-audit --no-fund && cd ../..
chmod +x v2/dev/run_portfolio_demo.sh
v2/dev/run_portfolio_demo.sh
```

The launcher creates fresh run and queue roots under the system temporary directory, explicitly selects local adapters, disables paid provider flags and removes `ANTHROPIC_API_KEY` from child processes. It skips the legacy import, seeds one generated `demo_fast` run, and binds Vite and Uvicorn to `127.0.0.1`. It prints the dashboard URL and data path; copy the URL into a browser. No cloud account, API key, model weights or match footage is needed. Ctrl+C stops the services. The generated directory remains available for inspection; delete only that specific directory after you no longer need it.

Set the capture to 1920×1080 at 100% browser zoom. Hide bookmarks, browser extensions, notification banners and personal profile UI. Open **Analysis runs**, select **Generated local workflow demo**, and use the storyboard questions verbatim. Keep the synthetic profile label, mode badge and captions visible. When showing a trace, expand **Tool execution trace** only after the response arrives. For a phone preview, check the final video at 375 px wide with the sound off; captions, mode and synthetic label must still be readable. The [architecture PNG](architecture-current-1920x1080.png) is the final slide.

The existing generated-fixture browser spec (`v2/dashboard/tests/e2e/agent-copilot.spec.ts`) is a repeatable smoke path: it uploads a newly generated video, waits for the worker, checks citations and actual trace names, verifies the pass question abstains, and asserts no console errors. It is separate from the launch script and stores its ignored screenshot under `v2/dashboard/test-results/`.

Three 1920×1080 Chromium screenshots from the isolated launcher were captured and visually inspected on 2026-11-05: [first screen](captures/tier-a-first-screen.png), [supported answer and executed tool](captures/tier-a-supported.png), and [pass-statistic abstention](captures/tier-a-abstention.png). They show only the generated run and no credentials, accounts or private footage. **No video recording exists yet.** For a phone-sized final video, crop or zoom each UI shot around the copilot; a full 1920-pixel screen reduced to 375 pixels makes text too small.

## Screenshot capture and privacy checklist

- Capture the first screen with generated match name, succeeded status and `Synthetic demo_fast · no YOLO inference` visible.
- Capture the supported answer with `Synthetic output`, `OFFLINE SCRIPTED AGENT`, a readable citation and an open trace showing the executed tool name.
- Capture the unsupported pass question with abstention and `No tool ran`.
- Capture disabled/error states only from a local generated run, if used in the edit.
- Review every frame at full resolution and 375 px preview. Remove accidental tabs, account names, notifications, credentials, private run IDs, source filenames and footage.
- Include a `LOCAL DEMO · SYNTHETIC FIXTURE` caption on every UI shot. Never show run `87da5c1b` or its artifacts in a public cut.
- Store only inspected, privacy-safe captures in the final package; do not commit raw video or private run data.

## CV-ready description

**Implemented:** Built a FastAPI and React video-analysis platform with asynchronous worker execution, provider-neutral repository/object-storage/queue ports, versioned SHA-256-checked artifacts, and a local evidence-grounded copilot. Added a bounded read-only agent with five approved tools, server-rendered citations, traceability and deterministic regression evaluation. Real YOLOv8m and ByteTrack execution was validated locally on private footage; Azure staging previously exercised the synthetic pipeline. **Future scope:** human-reviewed agent labels, measured live Anthropic tool selection and rights-cleared public real-footage demo.

## Claim ledger for the 60-second cut

| Caption | Source or repeatable verification |
|---|---|
| Async queue and worker | `v2/src/footballai_v2/execution/coordinator.py`, `worker.py`; generated fixture browser flow |
| Immutable run and artifact checks | `contracts/v1/analysis_run.py`, `execution/executor.py`, `storage/local_analysis_runs.py`; run detail artifact panel |
| Synthetic, no YOLO | `execution/adapters/demo_pipeline.py`, `copilot/evidence.py`, `cli/seed_demo_workflow.py`; profile and output-nature badges |
| Claims and citations | `copilot/grounding.py`, `copilot/evidence.py`; browser citation assertion |
| Scripted tool trace | `copilot/evaluation/agent_models.py`, `copilot/agent/tools.py`, `components/AgentTraceView.tsx`; browser tool-name assertion |
| Pass abstention | `copilot/grounding.py` router; browser unsupported-question assertion |

## Remaining before final recording

Complete final CI and security gates, and record the video privately. Obtain footage rights only if upgrading to Tier B. Complete independent labels and measured live evaluation only if upgrading to Tier C. Publication remains a separate owner decision.
