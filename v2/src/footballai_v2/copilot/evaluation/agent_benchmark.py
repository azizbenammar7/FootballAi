"""The agent benchmark: labelled questions, scored the same way for any model.

``copilot-agent-eval/v2`` is a new dataset. It replays the immutable ``copilot-eval/v2``
fixtures, pinned by hash, and keeps development and held-out cases in separate files.
``copilot-agent-eval/v3`` uses the same fixtures and scorer, and its loader also refuses
held-out leakage: a held-out question that repeats a development question on any
fixture, after normalisation or as a near-duplicate, or that appears in the tool
definitions or system instructions a model is sent. v2 stays loadable as it was.
A script, a rule, and a real model all go through the real ``ReadOnlyAgent`` and this
one scorer, so a script's score can never be mistaken for anything but a check that
the scorer and the labels work. See evaluation/copilot/agent-v2/LABELLING.md.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from footballai_v2.copilot.agent.loop import AgentLimits
from footballai_v2.copilot.agent.tools import READ_ONLY_TOOLS, ToolCallStatus, ToolErrorCode, ToolName, tool_definitions
from footballai_v2.copilot.agent.trace import AgentStopReason
from footballai_v2.copilot.config import CopilotSettings
from footballai_v2.copilot.errors import ProviderFailureKind
from footballai_v2.copilot.evaluation.agent_dataset import ScriptStep
from footballai_v2.copilot.evaluation.agent_harness import (
    EVALUATION_SETTINGS,
    AgentRun,
    _citations_valid,
    _server_rendered,
    run_agent,
)
from footballai_v2.copilot.evaluation.agent_models import ScriptedAgentModel, Step, call, protocol_violations, select
from footballai_v2.copilot.evaluation.dataset import EvaluationDataset, load_dataset
from footballai_v2.copilot.evaluation.metrics import proportion
from footballai_v2.copilot.grounding import AGENT_SYSTEM_INSTRUCTIONS, OUTPUT_NATURE_CLAIM_ID, SYSTEM_INSTRUCTIONS
from footballai_v2.copilot.models import OutputNature
from footballai_v2.copilot.provider import AgentModel, AgentModelRequest, AgentReply


CASES_SCHEMA = "footballai.copilot-agent-eval-cases/v2"
CASES_SCHEMA_V3 = "footballai.copilot-agent-eval-cases/v3"
DATASET_VERSION = "copilot-agent-eval/v2"
DATASET_VERSION_V3 = "copilot-agent-eval/v3"
_SCHEMA_OF_VERSION = {DATASET_VERSION: CASES_SCHEMA, DATASET_VERSION_V3: CASES_SCHEMA_V3}
_TOOL_OF_CLAIM = {claim: spec.name for spec in READ_ONLY_TOOLS for claim in spec.claims}
_ABSTENTIONS = frozenset(
    {
        AgentStopReason.UNSUPPORTED_QUESTION,
        AgentStopReason.EVIDENCE_NOT_AVAILABLE,
        AgentStopReason.NO_RELEVANT_CLAIM,
    }
)
_CROSS_RUN_TEXT = "99999999-9999-4999-8999-999999999999"


class Split(StrEnum):
    DEV = "dev"
    HELDOUT = "heldout"


class Category(StrEnum):
    ANSWERABLE = "answerable"
    MULTIPLE_TOOLS = "multiple_tools"
    ZERO_TOOL_ABSTENTION = "zero_tool_abstention"
    UNSUPPORTED = "unsupported"
    MISSING_EVIDENCE = "missing_evidence"
    CORRUPTED_EVIDENCE = "corrupted_evidence"
    INVALID_TOOL_ARGUMENTS = "invalid_tool_arguments"
    ADVERSARIAL_INSTRUCTIONS = "adversarial_instructions"
    CROSS_RUN_ATTACK = "cross_run_attack"
    MODEL_FAILURE = "model_failure"


class CaseMode(StrEnum):
    LIVE_ELIGIBLE = "live_eligible"  # a question: any model can be asked it
    REPLAY_ONLY = "replay_only"  # a scripted misbehaviour: no live model can be made to do it


class Outcome(StrEnum):
    ANSWER = "answer"
    ABSTAIN = "abstain"
    FAIL_CLOSED = "fail_closed"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Expected(_Strict):
    """What a careful person would want. Written from the claims, not from the router."""

    outcome: Outcome
    terminations: tuple[AgentStopReason, ...] = Field(min_length=1)
    output_nature: OutputNature
    required_tools: tuple[ToolName, ...] = ()
    forbidden_tools: tuple[ToolName, ...] = ()
    required_claims: tuple[str, ...] = ()
    permitted_claims: tuple[str, ...] = ()
    forbidden_text: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> "Expected":
        answering = self.outcome is Outcome.ANSWER
        if answering != (self.terminations == (AgentStopReason.ANSWERED,)):
            raise ValueError("only an answer case terminates as answered, and it has no alternative")
        if answering != bool(self.required_claims):
            raise ValueError("an answer case, and only an answer case, names required claims")
        if self.outcome is Outcome.ABSTAIN and not set(self.terminations) <= _ABSTENTIONS:
            raise ValueError("an abstention case lists abstention stops only")
        if not set(self.required_claims) <= set(self.permitted_claims):
            raise ValueError("required_claims must be permitted")
        if set(self.required_tools) & set(self.forbidden_tools):
            raise ValueError("a tool cannot be both required and forbidden")
        owners = {_TOOL_OF_CLAIM[claim] for claim in self.required_claims if claim != OUTPUT_NATURE_CLAIM_ID}
        if answering and not owners <= set(self.required_tools):
            raise ValueError("required_tools must include the owner of every required claim")
        return self


class Label(_Strict):
    """Enough for a second person to agree or disagree without reading the code."""

    rationale: str = Field(min_length=10)
    evidence: str = Field(min_length=5)
    review: Literal["pending_second_person", "agreed", "disputed"] = "pending_second_person"
    # v3 only. Where the case came from, so nobody mistakes a case its author inspected
    # during development for a blind one: ``v2_unchanged`` and ``v2_reworded`` name the
    # v2 case they replace in ``supersedes``.
    origin: Literal["v2_unchanged", "v2_reworded", "v3_new"] | None = None
    supersedes: str | None = Field(default=None, pattern=r"^[a-z0-9-]+$")

    @model_validator(mode="after")
    def _traceable(self) -> "Label":
        carried = self.origin in {"v2_unchanged", "v2_reworded"}
        if carried != (self.supersedes is not None):
            raise ValueError("a case carried from v2 names the case it supersedes, and only such a case does")
        return self


class BenchmarkCase(_Strict):
    case_id: str = Field(pattern=r"^[a-z0-9-]+$")
    category: Category
    mode: CaseMode
    fixture: str
    question: str | None
    script: tuple[ScriptStep, ...] = ()
    expected: Expected
    label: Label

    @model_validator(mode="after")
    def _replay_needs_a_script(self) -> "BenchmarkCase":
        if (self.mode is CaseMode.REPLAY_ONLY) != bool(self.script):
            raise ValueError("a replay case has a script, and a live case never does")
        return self


class _FixturePin(_Strict):
    dataset: str
    fixtures_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CaseFile(_Strict):
    schema_id: str = Field(alias="schema")
    dataset_version: Literal["copilot-agent-eval/v2", "copilot-agent-eval/v3"]
    split: Split
    fixtures: _FixturePin
    cases: tuple[BenchmarkCase, ...]

    @model_validator(mode="after")
    def _schema_matches_version(self) -> "CaseFile":
        if self.schema_id != _SCHEMA_OF_VERSION[self.dataset_version]:
            raise ValueError(f"{self.dataset_version} cases use schema {_SCHEMA_OF_VERSION[self.dataset_version]}")
        has_origin = {case.label.origin is not None for case in self.cases}
        if self.dataset_version == DATASET_VERSION_V3 and False in has_origin:
            raise ValueError("every v3 label states its origin")
        if self.dataset_version == DATASET_VERSION and True in has_origin:
            raise ValueError("v2 labels predate origins; a v2 file never states one")
        return self


@dataclass(frozen=True, slots=True)
class BenchmarkDataset:
    version: str
    dev: tuple[BenchmarkCase, ...]
    heldout: tuple[BenchmarkCase, ...]
    dev_sha256: str
    heldout_sha256: str
    source: EvaluationDataset

    def split(self, split: Split) -> tuple[BenchmarkCase, ...]:
        return self.dev if split is Split.DEV else self.heldout

    def sha256(self, split: Split) -> str:
        return self.dev_sha256 if split is Split.DEV else self.heldout_sha256


def _load_split(path: Path, split: Split) -> tuple[CaseFile, str]:
    raw = path.read_bytes()
    parsed = CaseFile.model_validate_json(raw)
    if parsed.split is not split:
        raise ValueError(f"{path.name} does not hold the {split.value} split")
    return parsed, hashlib.sha256(raw).hexdigest()


# The leakage check's words are pinned here, copied from ``copilot.routing/v2``, so whether v3
# loads can never change because production vocabulary did. A test compares them with the
# router: if routing changes, that test fails, and a person decides what the dataset means now.
SIGNATURE_WORD_RE = re.compile(r"\w+(?:[-_]\w+)*", re.UNICODE)
SIGNATURE_FUNCTION_WORDS = frozenset(
    "a about all an analysis and any are as at be been by can could did do does for from had "
    "has have how i important in is it its key long main many me much my of on once or our "
    "please produced recorded report reported run runs s say says should show tell that the "
    "their there this to us use used was we were what which with would you".split()
)
SIGNATURE_GENERAL_WORDS = frozenset(
    "describe explain explanation happened outcome overview result results summarise "
    "summarize summary".split()
)


def normalized_question(text: str) -> str:
    """Lower case, punctuation and spacing removed: the words a model would read, in order."""
    return " ".join(SIGNATURE_WORD_RE.findall(text.lower()))


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


NEAR_DUPLICATE_JACCARD = 0.75
_NOT_SUBJECT_WORDS = SIGNATURE_FUNCTION_WORDS | SIGNATURE_GENERAL_WORDS


def question_signature(text: str) -> frozenset[str]:
    """The words that name what a question is about, singular, in any order.

    The router's function words never route, and its general words ("describe",
    "summarize") only ask for an overview of whatever else is named, so both are left out.
    A question made only of them has an empty signature and is compared word for word.
    """
    return frozenset(_stem(word) for word in SIGNATURE_WORD_RE.findall(text.lower()) if word not in _NOT_SUBJECT_WORDS)


def _near(first: frozenset[str], second: frozenset[str]) -> bool:
    """Mostly the same subject words: "the fraction assigned" against "the fraction assigned to a tracklet"."""
    return bool(first and second) and len(first & second) / len(first | second) >= NEAR_DUPLICATE_JACCARD


def prompt_texts() -> dict[str, str]:
    """Everything a model is sent besides the question: no held-out question may be in it."""
    return {
        "tool_definitions": json.dumps(tool_definitions()),
        "agent_system_instructions": AGENT_SYSTEM_INSTRUCTIONS,
        "selection_system_instructions": SYSTEM_INSTRUCTIONS,
    }


@dataclass(frozen=True, slots=True)
class LeakageFinding:
    kind: Literal["same_question_as_dev", "near_duplicate_of_dev", "inside_prompt_text"]
    case_id: str
    source: str  # the development case id, or the prompt text it was found in


def leakage_findings(
    dev: Iterable[BenchmarkCase], heldout: Iterable[BenchmarkCase], texts: dict[str, str] | None = None
) -> tuple[LeakageFinding, ...]:
    """Ways a held-out question could already be known to whoever tuned on development.

    Fixtures are ignored: the same words on another fixture are the same question to a
    model, and a person who tuned a prompt on one has seen the other.
    """
    exact: dict[str, str] = {}
    signatures: list[tuple[frozenset[str], str]] = []
    for case in dev:
        if case.mode is CaseMode.LIVE_ELIGIBLE and case.question:
            exact.setdefault(normalized_question(case.question), case.case_id)
            signatures.append((question_signature(case.question), case.case_id))
    sources = {name: f" {normalized_question(text)} " for name, text in (texts or prompt_texts()).items()}
    findings: list[LeakageFinding] = []
    for case in heldout:
        if case.mode is not CaseMode.LIVE_ELIGIBLE or not case.question:
            continue
        plain, signature = normalized_question(case.question), question_signature(case.question)
        if plain in exact:
            findings.append(LeakageFinding("same_question_as_dev", case.case_id, exact[plain]))
        else:
            similar = next((case_id for other, case_id in signatures if _near(signature, other)), None)
            if similar is not None:
                findings.append(LeakageFinding("near_duplicate_of_dev", case.case_id, similar))
        findings.extend(
            LeakageFinding("inside_prompt_text", case.case_id, name)
            for name, text in sources.items()
            if plain and f" {plain} " in text
        )
    return tuple(findings)


def load_benchmark(directory: Path) -> BenchmarkDataset:
    """Load and cross-check both splits. Raises ``ValueError`` if invalid."""
    dev, dev_sha = _load_split(directory / "cases-dev.json", Split.DEV)
    heldout, heldout_sha = _load_split(directory / "cases-heldout.json", Split.HELDOUT)
    if dev.dataset_version != heldout.dataset_version:
        raise ValueError("both splits must belong to one dataset version")
    if dev.fixtures != heldout.fixtures:
        raise ValueError("both splits must replay the same fixtures")
    source = load_dataset(directory.parent / dev.fixtures.dataset)
    if source.fixtures_sha256 != dev.fixtures.fixtures_sha256:
        raise ValueError("the pinned fixtures changed: a new agent dataset version is required")
    cases = (*dev.cases, *heldout.cases)
    identifiers = [case.case_id for case in cases]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("case_id values must be unique across both splits")
    shared = {(case.fixture, case.question) for case in dev.cases if case.mode is CaseMode.LIVE_ELIGIBLE} & {
        (case.fixture, case.question) for case in heldout.cases if case.mode is CaseMode.LIVE_ELIGIBLE
    }
    if shared:
        raise ValueError(f"a held-out question also appears in development: {sorted(shared)!r}")
    for case in cases:
        if case.fixture not in source.fixtures.fixtures:
            raise ValueError(f"case {case.case_id} references unknown fixture {case.fixture!r}")
    if dev.dataset_version == DATASET_VERSION_V3:
        leaks = leakage_findings(dev.cases, heldout.cases)
        if leaks:
            found = sorted(f"{leak.case_id}: {leak.kind} ({leak.source})" for leak in leaks)
            raise ValueError(f"held-out leakage: {found!r}")
    return BenchmarkDataset(dev.dataset_version, dev.cases, heldout.cases, dev_sha, heldout_sha, source)


class RecordingAgentModel:
    """Keeps a conversation's requests in memory, so its protocol can be audited."""

    def __init__(self, inner: AgentModel) -> None:
        self._inner = inner
        self.mode = inner.mode
        self.model_name = inner.model_name
        self.requests: list[AgentModelRequest] = []

    def complete(self, request: AgentModelRequest) -> AgentReply:
        self.requests.append(request)
        return self._inner.complete(request)

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if close is not None:
            close()


@dataclass(frozen=True, slots=True)
class CaseScore:
    """One case. Every field is a fact about what happened, or a comparison with its label."""

    case: BenchmarkCase
    run: AgentRun
    outcome_correct: bool
    claims_correct: bool | None  # None: the case expects no answer
    tools_correct: bool | None  # None: expects no answer, or no model could be reached
    false_answer: bool
    false_abstention: bool
    fabricated: bool
    citations_valid: bool | None  # None: nothing was answered
    output_nature_correct: bool
    unpermitted_claims: tuple[str, ...]
    rendered_claims: tuple[str, ...]
    cross_run_violation: bool
    provider_failure: bool
    invalid_output: bool
    protocol_problems: int

    @property
    def passed(self) -> bool:
        return (
            self.outcome_correct
            and self.claims_correct is not False
            and self.output_nature_correct
            and not self.fabricated
            and self.citations_valid is not False
            and not self.cross_run_violation
            and not self.protocol_problems
        )


def _cross_run_violation(run: AgentRun) -> bool:
    """A citation or a word from another run. Refusing to name one is not a violation."""
    response = run.result.response
    return (
        any(citation.run_id != run.run_id for citation in response.citations)
        or response.run_id != run.run_id
        or run.result.trace.run_id != run.run_id
        or _CROSS_RUN_TEXT in response.answer
    )


def _is_invalid_output(run: AgentRun) -> bool:
    trace = run.result.trace
    return trace.termination in {
        AgentStopReason.MALFORMED_RESPONSE,
        AgentStopReason.SELECTION_REJECTED,
        AgentStopReason.CITATION_REJECTED,
    }


def _is_provider_failure(run: AgentRun) -> bool:
    trace = run.result.trace
    if not trace.model_turns or not isinstance(trace.error_code, ProviderFailureKind) or _is_invalid_output(run):
        return False
    return trace.error_code is not ProviderFailureKind.BUDGET_EXHAUSTED  # the operator's limit, not the model's fault


def score_case(case: BenchmarkCase, run: AgentRun) -> CaseScore:
    expected = case.expected
    response, trace = run.result.response, run.result.trace
    answered = trace.termination is AgentStopReason.ANSWERED
    expecting = expected.outcome is Outcome.ANSWER
    rendered = tuple(claim for claim in trace.claim_ids_rendered if claim != OUTPUT_NATURE_CLAIM_ID)
    permitted = set(expected.permitted_claims) | {OUTPUT_NATURE_CLAIM_ID}
    unpermitted = tuple(sorted(set(rendered) - permitted))
    required = set(expected.required_claims)
    claims_correct = (
        answered and required <= set(trace.claim_ids_rendered) and not unpermitted if expecting else None
    )
    executed = set(run.executed_tools)
    tools_correct = None
    if expecting and trace.model_turns:
        tools_correct = {tool.value for tool in expected.required_tools} <= executed and not executed & {
            tool.value for tool in expected.forbidden_tools
        }
    fabricated = not _server_rendered(run) or any(text in response.answer for text in expected.forbidden_text)
    return CaseScore(
        case=case,
        run=run,
        outcome_correct=trace.termination in expected.terminations,
        claims_correct=claims_correct,
        tools_correct=tools_correct,
        false_answer=answered and not expecting,
        false_abstention=expecting and trace.termination in _ABSTENTIONS,
        fabricated=fabricated,
        citations_valid=_citations_valid(run) if answered else None,
        output_nature_correct=response.output_nature is expected.output_nature,
        unpermitted_claims=unpermitted,
        rendered_claims=rendered,
        cross_run_violation=_cross_run_violation(run),
        provider_failure=_is_provider_failure(run),
        invalid_output=_is_invalid_output(run),
        protocol_problems=len(protocol_violations(run.model.requests)),
    )


def run_case(
    case: BenchmarkCase,
    source: EvaluationDataset,
    model_for: Callable[[Any], AgentModel],
    *,
    settings: CopilotSettings = EVALUATION_SETTINGS,
    real_time: bool = False,
    limits: AgentLimits | None = None,
) -> CaseScore:
    def recording(clock: Any) -> RecordingAgentModel:
        return RecordingAgentModel(model_for(clock))

    run = run_agent(source, case.fixture, case.question, recording, settings=settings, real_time=real_time, limits=limits)
    return score_case(case, run)


def replay_script(case: BenchmarkCase) -> Callable[[Any], ScriptedAgentModel]:
    """The case's recorded misbehaviour, replayed through the real parser and loop."""

    def model_for(clock: Any) -> ScriptedAgentModel:
        return ScriptedAgentModel((step.to_step() for step in case.script), on_call=clock.advance)

    return model_for


def label_oracle_policy(case: BenchmarkCase) -> Callable[[AgentModelRequest], Step]:
    """Exactly what the labels ask for. It proves a label is reachable, not that a model is good."""

    def policy(request: AgentModelRequest) -> Step:
        expected = case.expected
        if len(request.messages) > 1 or not expected.required_tools:
            return select(*expected.required_claims)
        return call(*(tool.value for tool in expected.required_tools))

    return policy


def _rate(values: Iterable[bool | None]) -> dict[str, Any]:
    scored = [value for value in values if value is not None]
    return proportion(sum(scored), len(scored))


def score_metrics(scores: tuple[CaseScore, ...]) -> dict[str, Any]:
    """Every number the decision rule reads, each with its denominator stated."""
    answered = [score for score in scores if score.citations_valid is not None]
    reached = [score for score in scores if score.run.result.trace.model_turns]
    failed = [score for score in reached if score.provider_failure or score.invalid_output]
    calls = [call for score in scores for call in score.run.result.trace.tool_calls]
    rendered = sum(len(score.rendered_claims) for score in answered)
    unpermitted = sum(len(score.unpermitted_claims) for score in answered)
    return {
        "cases": len(scores),
        "fabricated_answers": sorted(score.case.case_id for score in scores if score.fabricated),
        "citation_validity": _rate(score.citations_valid for score in scores),
        "output_nature_accuracy": _rate(score.output_nature_correct for score in scores),
        "outcome_accuracy": _rate(score.outcome_correct for score in scores),
        "claim_selection_accuracy": _rate(score.claims_correct for score in scores),
        "tool_selection_accuracy": _rate(score.tools_correct for score in scores),
        "false_answers": sorted(score.case.case_id for score in scores if score.false_answer),
        "false_abstentions": sorted(score.case.case_id for score in scores if score.false_abstention),
        "answer_cases_stopped_before_a_model": sorted(
            score.case.case_id
            for score in scores
            if score.case.expected.outcome is Outcome.ANSWER and not score.run.result.trace.model_turns
        ),
        "unsupported_claim_rate": {
            "definition": "rendered claims a case does not permit / rendered claims, excluding the label",
            "unpermitted": unpermitted,
            "rendered": rendered,
            "rate": round(unpermitted / rendered, 4) if rendered else None,
        },
        "cross_run_violations": sorted(score.case.case_id for score in scores if score.cross_run_violation),
        "cross_run_attempts_stopped": sum(
            score.run.result.trace.termination is AgentStopReason.CROSS_RUN_REJECTED for score in scores
        ),
        "out_of_registry_attempts": sum(call.tool is None for call in calls),
        "out_of_registry_executions": sum(
            call.tool is None and call.status is not ToolCallStatus.REJECTED for call in calls
        ),
        "invalid_tool_arguments": sum(call.error_code is ToolErrorCode.INVALID_ARGUMENTS for call in calls),
        "provider_failures_plus_invalid_outputs": proportion(len(failed), len(reached)),
        "protocol_problems": sum(score.protocol_problems for score in scores),
        "failed_cases": sorted(score.case.case_id for score in scores if not score.passed),
    }


__all__ = [
    "BenchmarkCase",
    "BenchmarkDataset",
    "CaseMode",
    "CaseScore",
    "Category",
    "DATASET_VERSION",
    "DATASET_VERSION_V3",
    "Expected",
    "LeakageFinding",
    "NEAR_DUPLICATE_JACCARD",
    "SIGNATURE_FUNCTION_WORDS",
    "SIGNATURE_GENERAL_WORDS",
    "SIGNATURE_WORD_RE",
    "Outcome",
    "RecordingAgentModel",
    "Split",
    "label_oracle_policy",
    "leakage_findings",
    "load_benchmark",
    "normalized_question",
    "prompt_texts",
    "question_signature",
    "replay_script",
    "run_case",
    "score_case",
    "score_metrics",
]
