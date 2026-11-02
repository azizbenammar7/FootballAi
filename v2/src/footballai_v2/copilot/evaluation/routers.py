"""The question routing each dataset version was labelled against.

A dataset's labels only mean something under the routing they were written for.
``copilot-eval/v1`` and the decision rule accepted on it depend on the routing
below, so it is frozen here, verbatim, and replayed through the real service. It
is evaluation data in code form: the API and worker never import this package.

Only routing is versioned. Every claim v1 can offer keeps its exact text, and a
claim added later is not in any v1 route, so v1 cannot see it.
"""

from __future__ import annotations

import re

from footballai_v2.copilot.grounding import (
    ROUTING_VERSION,
    QuestionRoute,
    QuestionRouter,
    QuestionScope,
    route_question,
)


ROUTING_V1 = "copilot.routing/v1"

_RUN = "run"
_OUTPUT = "output"
_PROVENANCE = "provenance"
_ARTIFACTS = "artifacts"
_COUNTS = "counts"
_DETAIL = "detail"
_LIMITS = "limits"
_ALL_TOPICS = frozenset({_RUN, _OUTPUT, _PROVENANCE, _ARTIFACTS, _COUNTS, _DETAIL, _LIMITS})

_TOPIC_CLAIMS: dict[str, frozenset[str]] = {
    _RUN: frozenset({"run_status"}),
    _OUTPUT: frozenset({"output_nature"}),
    _PROVENANCE: frozenset({"provenance", "model_references", "source_duration"}),
    _ARTIFACTS: frozenset({"artifact_integrity"}),
    _COUNTS: frozenset({"diag_tracking"}),
    _DETAIL: frozenset(
        {
            "diag_assignment",
            "diag_duration",
            "diag_active",
            "diag_insufficient",
            "diag_interpretation",
            "diag_ground_truth",
            "diag_synthetic",
        }
    ),
    _LIMITS: frozenset({"missing_evidence", "boundaries"}),
}

_FUNCTION_WORDS = frozenset(
    "a about all an analysis and any are as at be been by can could did do does for from had "
    "has have how i in is it its long many me much my of on or our please produced recorded "
    "report reported run runs s say says show should tell that the their there this to use "
    "used was we were what which with would you".split()
)
_GENERAL_WORDS = frozenset(
    "describe explain explanation happened overview summarise summarize summary".split()
)
_TOPIC_WORDS: dict[str, tuple[str, ...]] = {
    **dict.fromkeys(
        "complete completed fail failed finished origin outcome partial pipeline profile result "
        "results state status succeed succeeded success successful version".split(),
        (_RUN,),
    ),
    **dict.fromkeys(
        "demo demo_fast executed fake genuine inference placeholder placeholders real synthetic "
        "unverified verified".split(),
        (_OUTPUT,),
    ),
    **dict.fromkeys(
        "checksum code commit hash length provenance repo repository revision sha sha-256 sha256 "
        "source video".split(),
        (_PROVENANCE,),
    ),
    **dict.fromkeys(("ml", "model", "models"), (_OUTPUT, _PROVENANCE)),
    **dict.fromkeys("artifact artifacts file files integrity passed registered verification".split(), (_ARTIFACTS,)),
    **dict.fromkeys("count counts detection detections total track tracklet tracklets tracks".split(), (_COUNTS,)),
    **dict.fromkeys(
        "active assigned assignment fraction fragmentation ground insufficient interpretation "
        "short simultaneous truth".split(),
        (_DETAIL,),
    ),
    **dict.fromkeys(("duration",), (_PROVENANCE, _DETAIL)),
    **dict.fromkeys("diagnostic diagnostics metric metrics tracked tracking".split(), (_COUNTS, _DETAIL)),
    **dict.fromkeys(
        "available boundaries caveat caveats evidence limit limitation limitations limits measured "
        "missing".split(),
        (_LIMITS,),
    ),
}
_KNOWN_WORDS = _FUNCTION_WORDS | _GENERAL_WORDS | _TOPIC_WORDS.keys()
_WORD_RE = re.compile(r"\w+(?:[-_]\w+)*", re.UNICODE)


def _claims(topics: frozenset[str] | set[str]) -> frozenset[str]:
    return frozenset(claim for topic in topics for claim in _TOPIC_CLAIMS[topic])


_GENERAL_ROUTE = QuestionRoute(QuestionScope.GENERAL, _claims(_ALL_TOPICS))
_UNSUPPORTED_ROUTE = QuestionRoute(QuestionScope.UNSUPPORTED, frozenset())


def route_question_v1(question: str | None) -> QuestionRoute:
    """Frozen with ``copilot-eval/v1``. Do not change it to improve a score."""
    text = (question or "").strip().lower()
    if not text:
        return _GENERAL_ROUTE
    words = _WORD_RE.findall(text)
    if not words or any(word not in _KNOWN_WORDS for word in words):
        return _UNSUPPORTED_ROUTE
    topics = {topic for word in words for topic in _TOPIC_WORDS.get(word, ())}
    if topics:
        return QuestionRoute(QuestionScope.SUPPORTED, _claims(topics))
    return _GENERAL_ROUTE if _GENERAL_WORDS.intersection(words) else _UNSUPPORTED_ROUTE


_ROUTERS: dict[str, tuple[str, QuestionRouter]] = {
    "copilot-eval/v1": (ROUTING_V1, route_question_v1),
    "copilot-eval/v2": (ROUTING_VERSION, route_question),
}


def routing_for(dataset_version: str) -> tuple[str, QuestionRouter]:
    """The routing a dataset was labelled against. Never a silent default."""
    try:
        return _ROUTERS[dataset_version]
    except KeyError:
        raise ValueError(f"no routing is bound to dataset {dataset_version!r}") from None


__all__ = ["ROUTING_V1", "route_question_v1", "routing_for"]
