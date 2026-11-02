# ADR-006: Agent endpoint and public trace

**Status:** Accepted for M2-C, pending owner review of the pull request
**Date:** 2026-11-02
**Deciders:** Aziz Ben Ammar

## Context

M2-B built a bounded agent that nothing could reach. M2-C puts it behind a local
endpoint and shows its trace in the dashboard, so a demo can prove which evidence tools
really ran. Three constraints shape this: the application has no authentication, no
model call may be paid for or implied, and a trace on screen must never contain a step
that did not happen.

## Decision

1. **A separate route.** `POST /api/v1/runs/{run_id}/agent` returns
   `{"explanation": ExplanationResponse, "trace": PublicAgentTrace}`. `/explain` keeps
   its contract, and the existing response model is reused unchanged.
2. **Two keys, and a late import.** The route needs `FOOTBALLAI_COPILOT_ENABLED=1`,
   `FOOTBALLAI_COPILOT_AGENT_ENABLED=1`, and a `local` or `test` environment. Otherwise
   it is not registered, and the agent and its adapter are not imported.
3. **Reachable from this machine only.** The gate proves what the server was told, not
   where it listens. The route also requires a loopback peer, a loopback `Host`, and an
   `Origin` that is absent or allowed. Anything else gets a plain `404`.
4. **No path to a paid model.** The route builds the scripted adapter, whatever the
   provider settings say. A real adapter needs a code change that passes
   `model_factory`, and that change belongs to M3 with its own approval.
5. **One adapter per request.** `ScriptedAgentModel` keeps every request it receives,
   so a shared instance would hold prompts for the life of the process.
6. **Two traces.** `AgentTrace` stays the operator's complete record and goes to the
   log. A client gets `PublicAgentTrace`, an explicit allow-list.
7. **The dashboard renders what the server returns.** It adds no steps, and its loading
   state names no tool.

## Options considered

### Response shape

| Option | For | Against | Decision |
|---|---|---|---|
| Nested `explanation` and `trace` | Existing contract and UI reused exactly | One level deeper | **Chosen** |
| Trace added to `ExplanationResponse` | Flat | Changes `/explain`'s contract, or makes the field optional | Rejected |
| Trace fetched separately | Small answer | Needs stored traces, and this design stores none | Rejected |

### What the client sees of a trace

| Option | For | Against | Decision |
|---|---|---|---|
| Public projection | Nothing is published by accident | A second schema to keep | **Chosen** |
| Return `AgentTrace` unchanged | No new code | Publishes provider failure classes and request ids, which `ProviderFailureKind` and `ProviderTelemetry` keep from clients, and shows the stub's constant token counts as if measured | Rejected |

### Offline adapter

| Option | For | Against | Decision |
|---|---|---|---|
| Import `route_following_policy` behind the gate | The demo runs the exact policy the baseline measures | The API imports one evaluation module when the gate is open | **Chosen** |
| Move the stub into the agent package | Keeps the import boundary simple | Churn, and the import-allow-list test would need widening | Rejected |
| Write a second policy for the demo | No import | Two policies could drift, and one would have no evidence | Rejected |

### Where the real provider setting leads

| Option | For | Against | Decision |
|---|---|---|---|
| Ignore it on this route | No paid path, and the label cannot be wrong | `/explain` and the agent can differ under one setting | **Chosen** |
| Follow it, as `/explain` does | Consistent, and ready for M3 | One environment variable would reach a never-run adapter from a button | Rejected for M2-C |

## A claim added beside this work

"What evidence is missing?" abstained on a complete run, saying the evidence did not
contain the answer, because `missing_evidence` exists only when something is missing.
That is misleading, and it is one of the demo's questions. M2-C adds
`no_missing_evidence`, offered only on that route, and only when diagnostics are valid,
every artifact passed integrity, and the run is not `unverified_or_partial`.

**`copilot.routing/v2` was deliberately not bumped.** The route's claim set grew, so a
complete run now answers where it used to abstain, under the same version string. A bump
would need a new dataset version under ADR-004, for a change no labelled case covers. The
`v1`, `v2`, and `agent-v1` baselines stay byte-identical, and a test keeps the claim out
of every other route. Traces from before and after this change are therefore not
comparable for this one question. The next routing change should bump the version and
add a labelled case.

## Consequences

- The dashboard can show a real tool trace, and no request can cost money.
- The scripted adapter is demonstrated on screen, so its label must stay visible. It
  proves plumbing, limits, and honesty, not that an agent chooses well.
- The route does not work through Compose, where the peer is a bridge address. Run it
  directly on `127.0.0.1`.
- A refused caller gets `404` for a well-formed POST, but a malformed body returns `422`
  and a wrong method `405`, because those are answered before the guard runs. That shows
  the route exists on an enabled server, and reaches no run.
- The local guard is not authentication. Hosted use still needs authentication,
  per-run authorization, and quotas.

## Action items

1. [x] Separate gate, route, public trace, dashboard mode, and browser validation.
2. [ ] M3: pass a real `model_factory` only with an approved evaluation plan.
3. [ ] M3: replace the dashboard's static offline banner when a real adapter exists. The
   banner is client text, while the badge on each answer comes from the server.
4. [ ] Add a labelled `no_missing_evidence` case at the next dataset version.
