# ADR-005: One bounded read-only agent

**Status:** Accepted for M2-B, pending owner review of the pull request
**Date:** 2026-11-02
**Deciders:** Aziz Ben Ammar

## Context

The copilot can select claims, but a person's question cannot yet decide what evidence
is fetched. M2-B adds that, without weakening ADR-003's core rule: a model never writes a
fact. No real model is available to this milestone, so everything must be testable
offline, and nothing may reach a public surface.

## Decision

Build one agent with a hand-written loop in `footballai_v2.copilot.agent`, separate from
FastAPI, inference, artifact generation, and Azure. It has no persistent memory.

1. **The model chooses tools and claim ids. Nothing else.**
2. **The router stays in front.** Unsupported questions cost nothing.
3. **Tools filter claims the server already built.** They hold no port, take no input,
   and return only claims eligible for this question.
4. **Retrieval is part of grounding.** Only ids a tool returned in this request can be
   rendered.
5. **Limits fail closed, with typed reasons.** There are no retries and no partial answers.
6. **A cross-run reference stops the request.** Other invalid calls get a fixed error and
   may recover, within budget.
7. **`AgentModel` is a seam.** The real adapter and the scripted stub share one reply
   parser, so a hostile reply exercises production code.

## Options considered

### Loop

| Option | Complexity | Dependency | Fit | Decision |
|---|---|---|---|---|
| Manual loop | Low to medium | None beyond `anthropic` | Every limit and failure is ours to test | **Chosen** |
| SDK Tool Runner | Low | Beta helper | Limits and traces would go through hooks | Rejected for now |
| LangGraph | Medium | Large | Solves branching workflows this agent lacks | Rejected |
| MCP server | Medium | New process | Solves sharing tools. There is one client | Rejected |
| Multi-agent | High | Varies | No task needs a second agent | Rejected |

The Tool Runner would save little: these tools are five filters, and the limits and
failure behaviour are the point of this work.

### Should a tool return every claim it owns, or only the question's?

| Option | For | Against | Decision |
|---|---|---|---|
| Only the question's | Keeps the rule that a model sees only eligible claims | The registry depends on routing | **Chosen** |
| Every claim, filter at the end | Tools look more independent | The model sees claims it may not select, and bad selections become likely | Rejected |

### Should an invalid tool call stop the request?

| Option | For | Against | Decision |
|---|---|---|---|
| Fixed `is_error` result, spend budget | A slip can recover, and it matches the target design | One more turn can be spent | **Chosen**, except for run references |
| Stop on any invalid call | Simplest to reason about | One slip costs the whole answer | Rejected |
| Stop on a run reference | That is an attempt to leave the bound run, not a slip | None worth keeping | **Chosen** for run references |

No path here produces an unsupported answer, because an error result carries no claims.

### Trade-off: parallel calls

Parallel tool calls stay on. They make a turn more useful, but one reply can then ask for
more than the budget allows. Such a turn is refused whole, before any tool runs, because
running part of it would make results depend on block order.

## Consequences

- **Easier:** M2-C only wires `ReadOnlyAgent` to a route. Each failure has a typed reason
  and a trace.
- **Easier:** CI drives every sequence, including hostile ones, for free.
- **Harder:** every new fact still needs a claim, citations, vocabulary, and a tool owner.
- **Harder:** the agent's value cannot show on v2, because the fake already scores
  perfectly there. Independent questions are needed.
- **Accepted:** a broad summary cannot use all five tools within the budget of four.
- **Accepted:** the SDK timeout cannot cut off a server dripping bytes. A late reply is
  discarded, but the wait is not interrupted.
- **Accepted:** `ExplanationService` and the agent stay separate entry points until
  evaluation shows which one should serve questions.

## Revisit when

- A live probe rejects strict tools with `output_config.format`.
- M3 shows that the router refuses questions the agent could answer safely.
- A second client must share these tools.

## Action items

1. [ ] Live probe, with approval, of strict tools and structured output together.
2. [x] M2-C endpoint, new dashboard reasons, and trace exposure.
3. [ ] Accept the agent decision rule before any paid agent run.
4. [x] Hard per-call deadline, on the client side, in M3-A. See ADR-007. Whether the
   provider stops work is unknown.
