# LinkedIn demo plan

| | |
|---|---|
| **Status** | Planned. Nothing has been recorded or published |
| **Owner** | Aziz Ben Ammar |
| **Date** | 2026-11-01 |
| **Next milestone** | M0 footage and a verified real run |
| **Milestone** | M5 in the [roadmap](AI_ENGINEERING_ROADMAP.md#m5--linkedin-demo--portfolio-release) |
| **Publishing** | Owner only, after the release gate passes |

## Current honest position

A recording made today would mislead or fail. Four things block it:

| Blocker | Evidence | Fix |
|---|---|---|
| The dashboard cannot ask a question | Fixed in M2-A, merged | M2-A |
| No shareable verified real run exists | Run `87da5c1b` is `verified_real`, but on broadcast footage that must never be published | Run on licensed footage, M0 |
| Footage rights are not established | No licence record exists | Owner, M0 |
| The flagship follow-up abstained | Fixed in M2-A, merged. It now explains fragmentation and what was not measured | M2-A |
| No language model has driven the agent | The trace viewer works, with an offline scripted adapter only | M1 and M3, each with approval |

The Tier C storyboard below needs a real model and evaluation results. The agent and
trace view exist, and a gated real path is in review in M3-A, but it has never reached a
real model. A recording today could only show the scripted adapter, which must stay
labelled. See the
[demo ladder](AI_ENGINEERING_ROADMAP.md#demo-ladder-and-fallbacks).

## Demo environments

| Environment | State | Use |
|---|---|---|
| **LOCAL DEMO** | Works on a laptop, bound to `127.0.0.1` | **Recommended** |
| **CLOUD DEMO** | Last evidence is old images, stopped PostgreSQL, and P6 partly applied. The copilot is forced off there | Do not record. Show CI and delivery evidence instead |
| **PRODUCTION-READY APPLICATION** | **Does not exist.** No authentication, authorization, or quotas | Never claim this |

The video must say "local demo" or equivalent. A public real-model endpoint without
authentication is not acceptable, so there is no hosted "try it" link. Offer the
video and the repository.

**Secure local setup:**

- Bind the API and dashboard to `127.0.0.1` only.
- Enable the copilot with `FOOTBALLAI_COPILOT_ENABLED=1` and
  `FOOTBALLAI_ENVIRONMENT=local`. The demo scripts do not set these.
- For real-model tiers, also set `FOOTBALLAI_COPILOT_PROVIDER=real` and
  `FOOTBALLAI_COPILOT_REAL_PROVIDER_ENABLED=1`, and run the API from a local Python
  environment with `v2/requirements-copilot-eval.txt` installed. No runtime image
  contains the SDK, so the Compose stack cannot serve these tiers.
- Load the key into one terminal that never appears on screen. Keep it out of
  shell history, files, and Git.
- Set a low monthly spend limit in the Anthropic Console.

## Target storyboard, Tier C

Target length is 80 seconds. Every step needs its "code needed" item finished first.

| Time | Screen | User action | System behaviour | Caption | Concept | Code needed |
|---|---|---|---|---|---|---|
| 0–8s | Title over the dashboard | None | None | "Most AI demos show what a model can say. This one shows what it refuses to say." | Problem | Video edit |
| 8–22s | Run Detail for a real run | Scroll to status and artifacts | Real labels, checksum-verified artifacts | "A real video, analysed by YOLOv8 and ByteTrack through a queue and worker." | Async CV pipeline | M0 run |
| 22–34s | Copilot panel | Ask "Summarize the most important verified results." | Tool calls appear, answer renders | "A read-only agent chooses what evidence to fetch." | Tool calling | M2, M4 trace |
| 34–46s | Citations | Open a citation | Field, artifact, and checksum shown | "Every sentence is written by the server, from verified evidence. Every one is cited." | Grounding | M4 readability |
| 46–56s | Copilot panel | Ask "How reliable was the tracking?" | Fragmentation claims, tracklet limits | "Track IDs are temporary. It says so." | Scientific honesty | M2-A vocabulary |
| 56–66s | Copilot panel | Ask "How many successful passes did player 7 make?" | Abstains | "Pass recognition doesn't exist here. So it doesn't answer." | Abstention | M2-A question box |
| 66–76s | Evaluation report | Show summary | Real measured results | "Tested against a deterministic baseline, including the cases it gets wrong." | LLMOps | M3 results |
| 76–88s | Architecture diagram, repository name | None | None | "Cloud-native, evaluated, read-only. Local demo." | Architecture | M5 diagram |

Follow-up questions are independent requests. Do not imply memory.

## Fallback storyboard, Tier A or B

Target length is 60 seconds. It needs the M0 run and M2-A. Tier B also needs M1 to
pass.

| Time | Screen | Action | Caption | Code needed |
|---|---|---|---|---|
| 0–8s | Title | None | "An AI copilot that can't invent a fact." | Video edit |
| 8–20s | Real run | Show labels and artifacts | "Real YOLOv8 and ByteTrack output, integrity-checked." | M0 run |
| 20–34s | Copilot | Ask a supported question | "Answers are built only from verified claims, with citations." | M2-A |
| 34–44s | Mode badge | Hold on the badge | Tier A: "This mode is deterministic. It is not a language model." Tier B: "Claude selects claims. It writes none of them." | None |
| 44–52s | Copilot | Ask an unsupported question | "No pass data exists, so no answer." | M2-A |
| 52–60s | Architecture diagram | None | "On GitHub, with its evaluation reports." | M5 diagram |

Say "open source" only after the licence decision.

## Required features by tier

| Feature | Exists | Tier A | Tier B | Tier C |
|---|---|---|---|---|
| Dashboard, run detail, artifacts | Yes | Yes | Yes | Yes |
| Explanation panel with badges and citations | Yes | Yes | Yes | Yes |
| Question box | Merged | Yes | Yes | Yes |
| Verified real run on shareable footage | **No** | Yes | Yes | Yes |
| `reliable` vocabulary | Merged | Optional | Optional | Yes |
| Live real model | **Never run** | No | Yes | Yes |
| Tool-using agent | Offline scripted, real path gated and unused | No | No | Yes, with a real model |
| Trace view | Merged | No | No | Yes |
| Evaluation summary shot | Deterministic only | Optional | Yes | Yes |

## Release gate

Run this immediately before recording, and again before publishing. Every item must
pass. Record the commit and date below.

**Product**

- [ ] Frontend works on the recorded path.
- [ ] Backend works on the recorded path.
- [ ] A successful run is reproducible from documented steps.
- [ ] Real and synthetic labels are correct on every screen.
- [ ] The copilot runs in its stated mode, and the badge matches.
- [ ] Evidence retrieval is verified for each question shown.
- [ ] Citations resolve and are readable.
- [ ] No unsupported statistic appears.
- [ ] Failure states are clear: disabled copilot, provider failure, limits reached.
- [ ] No browser-console errors.
- [ ] The interface is readable at 720p playback.
- [ ] Response time is acceptable, and matches the measured latency. The video may
      be cut only if the cut is not used to imply speed.

**Evidence**

- [ ] Fake-provider output is never called a real LLM.
- [ ] The scripted agent shows `OFFLINE SCRIPTED AGENT` in every shot, and is never
      called an AI agent.
- [ ] Synthetic analysis is never called real YOLO inference.
- [ ] Tracklets are never called verified players.
- [ ] Each claim is in the [claim register](#claim-register).
- [ ] Evaluation numbers match a committed report.

**Safety**

- [ ] No credentials, sensitive data, or personal data, per the checks below.
- [ ] No public endpoint was opened.
- [ ] Spend stayed within the approved budget.

**Reproduction**

- [ ] Screens and video can be regenerated from the documented steps.
- [ ] Two clean-start runs match.

| Check | Commit | Date | Result |
|---|---|---|---|
| Pre-recording | | | |
| Pre-publishing | | | |

## Claim register

Add a row for each claim. An empty proof means the claim stays out.

| Claim | Status today | Proof |
|---|---|---|
| Real YOLOv8m + ByteTrack ran on a full 94-minute match | Verified, local | `reports/v2-validation/5m-15m-45m-94m-comparison.md` |
| About 4.5 hours for that 95.65-minute video on Apple MPS | Verified, one run | Same report |
| 8,099 raw track IDs, 97.65% too short to score | Verified, one run | Same report |
| Copilot failed closed on 10 of 10 invalid replies | Verified, scripted cases | `evaluation/copilot/v1/baseline-report.json` |
| Fabricated answers 0 in the baseline | Verified, by construction, deterministic only | Same report |
| Citation validity 100% in the baseline | Verified, deterministic only | Same report |
| Deployed to Azure staging | Verified at P4, `demo_fast` | `docs/v2/azure-staging-architecture.md` |
| Any real-model result | **Not measured** | Pending M1 |
| Offline scripted agent runs real tools, limits, and grounding through a local API | Verified, scripted adapter only | [Browser validation](../v2/COPILOT_AGENT.md#browser-validation) |
| Any claim that an AI model chooses tools | **Not measured** | Pending M3-B |
| Hard client-side deadline on a model call | Verified against a local server only | `v2/tests/test_copilot_agent_sdk_protocol.py`. Does not show that a provider stops work |

## Recording requirements

- 1920×1080, 16:9, 30 fps, H.264 MP4.
- Fresh browser profile, with no extensions, bookmarks, or signed-in accounts.
- Browser zoom high enough to read at 720p. Aim for 125 to 150%.
- Burned-in captions, because many viewers watch muted.
- Short, accurate cuts. Keep real response time visible at least once.
- A soft, plain audio track or none. Any music must be licensed.
- A separate plain-language description for accessibility.

## Privacy checks

Inspect the finished video frame by frame, and every screenshot, for:

- [ ] Private tokens, API keys, or session cookies.
- [ ] Azure subscription, tenant, or resource identifiers.
- [ ] Personal email addresses and account names.
- [ ] Private URLs, including staging hostnames and SAS URLs.
- [ ] Internal logs and stack traces.
- [ ] User information.
- [ ] Paid API credentials or Console screens showing account details.
- [ ] Local filesystem paths and usernames. Check the shell prompt, editor title
      bars, and any manifest view.
- [ ] Browser tabs, notifications, and desktop content.
- [ ] Faces, shirt names, or other identifying details in the footage.

Check any run-detail or manifest view for input URIs before recording it.

## Footage and copyright

- Use only footage that may be shared publicly. Licensed broadcast footage is not
  acceptable, and fair-use reasoning is not a plan.
- **Best option:** a clip you film yourself at an amateur match, with permission
  from the club and players.
- **Second option:** an openly licensed clip, with its licence checked for that
  specific clip, and any attribution or commercial limits recorded.
- SoccerNet and YouTube clips named in the old README are not automatically
  shareable. Do not assume they are.
- The clip used for the earlier AI1 and AI2A work must not be shown unless its
  public-sharing terms are confirmed.
- Never commit raw match video or sensitive footage. Keep the licence record outside
  the public repository.
- Check licences for fonts, music, logos, and club crests that appear on screen.

## Architecture graphic

Two editable Mermaid sources are ready:

- [Current system overview](ARCHITECTURE_CURRENT.md#system-overview)
- [Target diagram](ARCHITECTURE_TARGET.md#target-diagram)

**No exported image exists yet.** Plan for M5:

1. Export SVG from the Mermaid source for the README.
2. Export a 1920×1080 PNG for the post and the video end card.
3. Check both by eye, in light and dark, at phone width.
4. Confirm the recruiter test: a stranger can explain the flow in a few seconds.
5. Show the target diagram publicly only with planned items clearly marked, or use
   the current diagram once it includes the validated agent.

## After recording

- Revise the [post draft](LINKEDIN_POST_DRAFT.md) from real results.
- Update the README and the demo link.
- The owner reviews, uploads, and publishes. No automation posts anything.
