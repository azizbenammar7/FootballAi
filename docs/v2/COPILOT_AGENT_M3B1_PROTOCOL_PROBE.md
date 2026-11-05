# M3-B1 plan: live protocol probe (not authorized)

| Status on 2026-11-04 | |
|---|---|
| Authorization | **None.** M3-B1 needs its own written approval from the owner. This document is a plan |
| Requests sent to Anthropic | **0.** No preflight, no generation |
| Prerequisites still open | Human label review (0 of 142), D7, the dataset choice below |
| What it measures | Whether the API accepts this agent's requests. **Not** model quality |

The approval gate and stage design are in [the decision rule](COPILOT_AGENT_DECISION_RULE.md#stages)
and the commands in [the M3-B runbook](COPILOT_AGENT_REAL_MODEL.md#m3-b-runbook). This plan
recalculates every pin, cap, and bound for dataset v3, and replaces the v2-era numbers.

## Scope

| Stage | Billed | Split | In M3-B1 | Needs rule accepted |
|---|---|---|---|---|
| A. Preflight: `models.retrieve` and input-token counts. No generation | No, but it sends the key to Anthropic | dev | **Yes**, with the owner's go-ahead | No |
| B. Protocol probe | Yes, at most 5 requests | dev | **Yes**, with its own approval record | No |
| C. Development evaluation | Yes | dev | No. Separate authorization | No |
| D. Held-out decision | Yes | held-out | No. Separate authorization | **Yes** |

Stage B answers one question: do the Messages API, the pinned SDK, `claude-haiku-5-5`, the
strict tool definitions, and the per-question output schemas work together on a real
account? It runs the first development questions in plan order until five requests are sent.
It never touches held-out, so it cannot shape a held-out label.

## Pins, recalculated

Computed offline on 2026-11-04 from PR #60's head `ac1473d`, clean tree, with
`current_provenance()` and `plan_benchmark()`. Nothing was sent anywhere.

| Pin | Value |
|---|---|
| Scorer | `agent-scorer/v2` |
| Instruction version | `copilot.agent-instructions/v1` |
| System instructions sha256 | `28dd6e8ac5c14c7f827715d9cc541c5d05af9b5bd2123cd63d44a5ccad666da1` |
| Tool definitions sha256 | `1cdf3ab4caa41efb6080ca6573ac82f8db474eb9cf89cc829ee911de23955f28` |
| SDK | `anthropic==2.3.0`, from `v2/requirements-copilot-eval.txt` |
| Fixtures sha256 (`v2` fixtures, used by v3) | `30b8efc2cbe99686d1e8d3862e3a9dcba2f83463f9239045e20c921c89806b7b` |
| `agent-v2` dev cases sha256 | `a18ad4dfffbeb167f35a4070df7d1c4982a99f15bc60cd6160ea89288ea2cff1` |
| `agent-v3` dev cases sha256 | `80351d6aaf0e1dca8523a688d8fbd1b5f726cd9408b49a0ccf28d8a2fbd1e8d6` |
| `agent-v3` held-out cases sha256 | `d0caff5fadcbca8a42a098ad64667d3793d346abee8dcc50b832ddf709688d71`. **Not frozen.** Changes if review changes a label |
| `agent-v2` rule v2 sha256 | `cacf59c8b6ac3c165467f65083d97699feea041439a864382407557e513136fa`, `proposed` |
| `agent-v3` rule | **None exists.** See [D7](COPILOT_AGENT_D7_DECISION.md#where-an-accepted-rule-must-live-two-findings) |
| Source commit | The commit the run starts from, on a clean tree. **Not fixed yet**: it will be a later `main` commit. Every merge changes it, and the CLI refuses a mismatch (`approval_other_commit`, `source_tree_modified`) |

Do not reuse the v2 held-out hash `2ca12538…` or the v2 caps of 24 questions and 120
requests. They describe a dataset that is no longer the decision split.

To recompute the pins offline on the day, from the commit you will run:

```bash
PYTHONPATH=v2/src python -c "import json; from footballai_v2.copilot.evaluation.agent_live import current_provenance; print(json.dumps(current_provenance(), indent=2))"
```

## Decision B1-1: which dataset the probe runs on

The benchmark CLI refuses any paid run on `agent-v3` with `dataset_not_frozen` until rule
files sit in that directory. So:

| | a. Probe on `agent-v2` dev now | b. Probe on `agent-v3` dev after the freeze |
|---|---|---|
| Possible with today's code | Yes | Only after D7 and the review |
| Questions | 68 dev cases. Content verified identical to v3 dev: only the `origin` and `supersedes` label fields differ | Same 68 |
| Hash pinned in the approval | `a18ad4df…` | `80351d6a…` |
| Rule hash pinned | `cacf59c8…` (proposed; a probe does not need it accepted) | The accepted v3 rule |
| Risk | None to held-out. It decouples protocol risk from governance | Protocol problems are found later |

The author leans to **b**, because it follows the agreed order (review, D7, then M3-B1).
**a** is valid if the owner wants protocol risk retired while the review runs. The owner decides.

## Request limits and cost upper bounds

Prices from `v2/src/footballai_v2/copilot/pricing.py`, verified on 2026-11-02:
`claude-haiku-5-5` $0.10 input and $0.50 output per million tokens. They match the
`claude-api` skill's cached model table (2026-10-06) for input and output. The live pricing
page was not fetched in this session: **re-check it on the day.**

A bound assumes every question uses all 5 model turns, each sending the full 24,576-character
prompt budget plus 2,000 tokens of API-added prompt, plus all earlier output, and producing
1,024 output tokens. Characters are counted as tokens, which overstates. These are ceilings,
not estimates: real questions should cost far less, and nothing here has been measured.

| Stage | Split | Questions reaching a model | Distinct output schemas | Max requests | Worst case, Haiku 5.5 | Proposed caps |
|---|---|---:|---:|---:|---:|---|
| B. Probe | dev | 31 planned, stopped at 5 requests | - | **5** (code limit) | **$0.0179** (5 × $0.00358) | 5 requests, **$0.02** |
| C. Development | dev | 31 | 19 | 155 | $0.523 | 155 requests, $0.60 |
| D. Decision | held-out v3 | **48** | 28 | **240** | **$0.810** | See blocker below |

For reference, `claude-sonnet-5-5` ($2 / $10) would bound B at $0.358, C at $10.46, and D at
$16.20. C and D exceed the code's `MAX_AGENT_COST_USD` ceiling of $5, so Sonnet 5.5 cannot be
approved for them without a code change.

**Blocker for Stage D, not M3-B1.** The approval schema caps `max_model_requests` at 200
(`MAX_AGENT_REQUESTS`). A v3 decision run can need up to 240. With a 200 cap, an unusually
long run could stop early and read `INCOMPLETE_RUN`, which cannot be retried, because the
decision report path is single-use. Options, for the owner, before Stage D: accept the
200 cap with that risk; lower `max_model_turns`; or raise the ceiling in a reviewed change.

## Approval record for Stage B

Copy `evaluation/copilot/agent-v2/live-approval.template.json` **outside the repository**.
Fill it on the day, from the commit you run. The owner fills `status`, `approved_by`, and
`approved_on`. Nobody else does.

| Field | Value |
|---|---|
| `stage`, `split` | `protocol_probe`, `dev` |
| `model` | `claude-haiku-5-5` |
| `max_model_requests` | `5` (the code refuses more for a probe) |
| `max_cost_usd` | `0.02` |
| `cases_sha256` | `a18ad4df…` for option a, `80351d6a…` for option b |
| `decision_rule_sha256` | The rule file in the `--dataset` directory |
| `scorer_version`, `instruction_version`, the two prompt and tool hashes, `sdk_version` | From the pins table, recomputed |
| `source_commit` | The full SHA of the commit you run from |

## Preflight requirements

All must hold before Stage A. Each is checked by the code, by the owner, or both.

| # | Requirement | Checked by |
|---|---|---|
| 1 | Written authorization for M3-B1, naming Stage A and Stage B | Owner |
| 2 | A local machine, not CI. The runner refuses when `CI` or `GITHUB_ACTIONS` is set | Code |
| 3 | A Python 3.13 venv from `requirements-test.txt` and `v2/requirements-copilot-eval.txt` | Owner |
| 4 | A clean checkout at the pinned commit | Code (`source_tree_modified`) |
| 5 | `ANTHROPIC_BASE_URL` unset, or exactly `https://api.anthropic.com` | Code |
| 6 | `ANTHROPIC_API_KEY` entered for the session only: never in a file, the repository, a shell history, or a log | Owner |
| 7 | A monthly spend limit set in the Anthropic Console | Owner |
| 8 | Stage A passes: the model is `active` on the account and reports structured outputs and effort, and token counting succeeds for all 31 planned first requests. It sends 1 `models.retrieve` and 31 `count_tokens` calls, which are free but rate limited | Code (exit 3 if not) |
| 9 | The approval record matches every pin | Code |
| 10 | The output path is new. A report is never replaced | Code (`output_exists`) |

## Stop conditions

| Condition | Effect | Enforced by |
|---|---|---|
| 5 requests sent | Stop, exit 4, report written | `BudgetGuard`, `probe_over_its_cap` |
| Next request's worst case would pass $0.02 | Stop before sending | `BudgetGuard` |
| Authentication, permission, billing, model unavailable, bad request, or rate limit | Stop at once | `_FATAL_KINDS` |
| 3 provider failures in a row | Stop | `MAX_CONSECUTIVE_FAILURES` |
| Ctrl-C or a runner error | Stop, and the report is still written | `run_benchmark_live` |
| Any approval pin differs | Refuse before any request | `check_approval` |
| Any refusal, truncation at `max_tokens`, or `400` | Stop the stage. Report it. Do not retry or move to Stage C | Owner |
| A key or secret in any output | Stop. Revoke the key | Owner |

## Commands, for the authorized day only

Not run. Option a shown. For option b, use `evaluation/copilot/agent-v3`.

```bash
PYTHONPATH=v2/src python -m footballai_v2.cli.copilot_agent_benchmark --dataset evaluation/copilot/agent-v2 --preflight --split dev
```

```bash
PYTHONPATH=v2/src python -m footballai_v2.cli.copilot_agent_benchmark --dataset evaluation/copilot/agent-v2 --allow-paid-calls --stage protocol_probe --approval /path/outside/repo/approval-b.json --split dev --max-requests 5 --max-cost-usd 0.02 --output /path/outside/repo/probe.json
```

## What M3-B1 can and cannot claim

| It can show | It cannot show |
|---|---|
| The account, model, SDK, strict tools, and output schemas work together | Any quality rate. Five requests are not a sample |
| Real token usage and latency for a few requests | That 8 seconds p95 holds |
| Whether thinking fits in 1,024 output tokens on these questions | Anything about held-out |

After Stage B, report what happened, including every failure. Stage C needs its own
approval, and Stage D needs the accepted rule.
