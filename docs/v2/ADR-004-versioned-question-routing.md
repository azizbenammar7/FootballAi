# ADR-004: Versioned question routing for the interactive copilot

**Status:** Accepted for M2-A, pending owner review of the pull request
**Date:** 2026-11-01
**Deciders:** Aziz Ben Ammar

## Context

M2-A lets a person type questions, which needs routing and claim fixes. The committed
`copilot-eval/v1` baseline and the accepted decision rule depend on the old routing,
because a test ties the published numbers to live code. Fixing routing in place would
have changed v1's results, including the three false answers the rule refers to.

## Decision

Routes now carry a set of claim ids instead of topics. Production uses
`route_question` (`copilot.routing/v2`). The old router is frozen in
`copilot/evaluation/routers.py`, which the API and worker never import, and each
dataset version is bound to its routing. `ExplanationService` receives the router as
a parameter, so evaluation still runs through the real service.

The output-nature label is always rendered, but it only counts as an answer when the
route names it.

## Options considered

| Option | For | Against |
|---|---|---|
| **A. Frozen router, bound by dataset (chosen)** | v1 reruns in place, M1 commands keep their meaning, small because only routing differs | Keeps old vocabulary in the repository |
| B. Reproduce v1 by checking out an old commit | No old code kept | CI cannot keep checking it, and M1 would run against outdated code |
| C. Update the v1 numbers | Simple | Breaks immutability and silently redefines the accepted rule |
| D. Keep topics and add exceptions | Small diff | Cannot express "this claim only", and leaves the label problem in place |

## Consequences

- v1 is byte-identical and its deterministic results are unchanged.
- A test pins a fingerprint of the frozen vocabulary, computed from `main` at
  `30984c3`. Mutation testing showed the cases alone would miss added words.
- Every claim v1 can offer must keep its exact text. Changing one needs a decision.
- Scores from v1 and v2 are different experiments. Only identically labelled cases
  may be compared, using `--compare-with`.
- The router is still lexical, not semantic. See the limits in
  [COPILOT_EXPLANATION.md](COPILOT_EXPLANATION.md).

## Action items

1. [x] Bind each dataset to its routing, and refuse unbound datasets.
2. [ ] Decide after M1 whether a real selector still adds value on v2. The
   deterministic router leaves no labelled v2 case for it to improve.
