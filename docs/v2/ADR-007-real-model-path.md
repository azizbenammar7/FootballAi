# ADR-007: A gated real-model path for the agent

**Status:** Proposed for M3-A, pending owner review of the pull request
**Date:** 2026-11-02
**Deciders:** Aziz Ben Ammar

## Context

ADR-006 left the agent route with no path to a paid model, and said a real adapter needs
its own change and approval. M3-A prepares that path without using it. No API key was
used, and no request left this machine.

Three things found while auditing shaped this decision:

- **The SDK's timeout is not a deadline.** Against a local server sending one byte at a
  time, a call with a 0.3 second timeout ran for 11 seconds.
- **The SDK silently follows `ANTHROPIC_BASE_URL`.** That variable was set in the
  environment this work ran in. A key would have gone wherever it pointed.
- **The agent's provider setting also opens `/explain`.** That route has no cost guard
  and no loopback guard.

## Decision

1. **A third key.** `FOOTBALLAI_COPILOT_AGENT_REAL_MODEL_ENABLED=1` is needed on top of
   the copilot gate, the agent gate, and the existing provider settings.
2. **Refuse to start, never fall back.** With that flag set, `real_agent_factory` lists
   every blocker, and the server does not start if there is one. It holds codes, never
   values.
3. **No caps, no calls.** `FOOTBALLAI_COPILOT_AGENT_MAX_REQUESTS` and
   `FOOTBALLAI_COPILOT_AGENT_MAX_COST_USD` have no defaults. An invalid or missing value
   is no cap, and no cap is refused.
4. **Hosted platforms are recognised for what they are.** The environment name is a
   string someone typed, so Azure Container Apps, App Service, Kubernetes, and CI markers
   are refused too.
5. **A key goes only to Anthropic, or to loopback for contract tests.** Any other
   `ANTHROPIC_BASE_URL` is refused.
6. **A hard client-side deadline.** The call runs on a worker thread. At its deadline the
   caller stops waiting, and closes the client.
7. **The guard reserves before it sends.** A request is admitted only if its worst case
   fits, including what is reserved by requests in flight.
8. **The real agent keeps `/explain` offline.** `build_provider` returns the fake while
   the real agent is on.
9. **The dashboard asks the server what drives the agent.** A new `GET
   /api/v1/copilot/agent` sits behind the same local guard.

## Options considered

### What happens when the real path is requested and blocked

| Option | For | Against | Decision |
|---|---|---|---|
| Refuse to start | Impossible to miss. Labels can never disagree with intent | A misconfigured local server does not start | **Chosen** |
| Fall back to the script | The server keeps working | A demo shows a script while its operator believes a model is running | Rejected |
| Fail each request | The server starts | The failure shows only when someone asks, on camera | Rejected |

### Hard deadline

| Option | For | Against | Decision |
|---|---|---|---|
| Worker thread, then close the client | Small. Measured: caller back in 0.31 s, server saw the close at 0.38 s, thread gone by 0.57 s, for a 0.3 s deadline | One thread per call. Proves nothing upstream | **Chosen** |
| Streaming with a check between events | Natural cancellation points | A rewrite of the adapter and its contract tests, and one slow event still blocks | Rejected for now |
| Keep the SDK timeout, discard late replies | No change | The wait is not bounded at all | Rejected |

### Where the budget lives

| Option | For | Against | Decision |
|---|---|---|---|
| In memory, per process | Simple, and exact for one run | A restart resets it, and several workers multiply it | **Chosen**, with the Console's spend limit as the backstop |
| A file or table | Survives restarts | New persistence for a local tool | Rejected |

## What this does not prove

- **Client-side abort is not cancellation upstream.** Closing a connection stops this
  process waiting. Whether Anthropic stops generating, or still bills, is unknown, so an
  aborted turn is charged at its worst case.
- **The loopback guard is not authentication.** Hosted use still needs authentication,
  per-run authorization, and quotas.
- **Nothing here has met the live API.** See
  [COPILOT_AGENT_REAL_MODEL.md](COPILOT_AGENT_REAL_MODEL.md#what-is-unverified-live).

## Consequences

- **Easier:** a real run needs environment settings and no code change.
- **Easier:** tests exercise the full path with the real SDK against 127.0.0.1.
- **Harder:** `CopilotSettings` has three new fields, and the trace schema moves to `v2`.
- **Accepted:** `/explain` stays offline while the real agent is on, so the two can
  differ on one server.
- **Accepted:** a closed connection leaves a brief daemon thread behind. A test shows it
  ends.
- **Accepted:** the scripted path still imports one evaluation module for its adapter, as
  ADR-006 chose. The real path imports none, and a test enforces it.

## Pre-existing issues, not changed here

- `/explain` with the real provider is local-only, but it has no loopback guard. This
  matters on a server bound to a shared address.
- The M1 paths (`AnthropicTextProvider`, `evaluation/live.py`) still follow
  `ANTHROPIC_BASE_URL`. **Check that variable before any M1 run.**

## Revisit when

- A live probe rejects strict tools with `output_config.format`.
- First-use schema compilation exceeds the 20 second turn limit.
- More than one process must share a budget.

## Action items

1. [x] Third key, blockers, caps, hard deadline, endpoint check, cost guard.
2. [x] Server-reported adapter, replacing the dashboard's static banner (ADR-006 item 3).
3. [ ] Owner accepts or changes the [decision rule](COPILOT_AGENT_DECISION_RULE.md).
4. [ ] Apply the endpoint check to the M1 paths.
