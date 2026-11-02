"""The closed registry of read-only tools the agent may call.

Tools select from the claims the server already built for one request. They
hold no reference to a repository, storage, or filesystem, take no input, and
cannot name a run: the registry is constructed for one run by the caller, so a
tool cannot be pointed anywhere else. There is deliberately no tool for SQL, a
shell, Azure, artifact mutation, video processing, or the network.

Whatever a tool returns is checked against the request's catalog before the
model sees it, so tool output can never introduce a claim of its own.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Literal, Mapping

from pydantic import Field, ValidationError

from footballai_v2.copilot.evidence import ValidatedRunEvidence
from footballai_v2.copilot.grounding import OUTPUT_NATURE_CLAIM_ID, EvidenceClaim
from footballai_v2.copilot.models import Citation, CopilotModel, OutputNature


class ToolName(StrEnum):
    GET_RUN_STATUS = "get_run_status"
    GET_TRACKING_DIAGNOSTICS = "get_tracking_diagnostics"
    GET_ARTIFACT_METADATA = "get_artifact_metadata"
    GET_MODEL_PROVENANCE = "get_model_provenance"
    GET_EVIDENCE_LIMITATIONS = "get_evidence_limitations"


class ToolCallStatus(StrEnum):
    OK = "ok"
    NO_EVIDENCE = "no_evidence"  # the tool ran; this run has nothing of it for this question
    REJECTED = "rejected"  # the call never ran
    FAILED = "failed"  # the tool ran and its result was not usable


class ToolErrorCode(StrEnum):
    """The only failure text a model ever sees for a tool. Exceptions never reach it."""

    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    DUPLICATE_CALL = "duplicate_call"
    TOOL_FAILED = "tool_failed"
    INVALID_TOOL_OUTPUT = "invalid_tool_output"
    CROSS_RUN_REFERENCE = "cross_run_reference"


class NoInput(CopilotModel):
    """Every approved tool's input: nothing. Any argument at all is rejected."""


class ToolClaim(CopilotModel):
    id: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=4000)
    citations: list[Citation] = Field(min_length=1)


class ToolResultPayload(CopilotModel):
    """What the model receives from a tool: server-authored claims, nothing else."""

    tool: ToolName
    status: Literal[ToolCallStatus.OK, ToolCallStatus.NO_EVIDENCE]
    output_nature: OutputNature
    claims: list[ToolClaim]


ToolHandler = Callable[[tuple[EvidenceClaim, ...]], tuple[EvidenceClaim, ...]]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: ToolName
    description: str
    claims: frozenset[str]
    handler: ToolHandler
    input_model: type[CopilotModel] = NoInput


def _claims_owned_by(owned: frozenset[str]) -> ToolHandler:
    def handler(catalog: tuple[EvidenceClaim, ...]) -> tuple[EvidenceClaim, ...]:
        return tuple(claim for claim in catalog if claim.claim_id in owned)

    return handler


def _tool(name: ToolName, description: str, claims: set[str]) -> ToolSpec:
    owned = frozenset(claims)
    return ToolSpec(name, description, owned, _claims_owned_by(owned))


READ_ONLY_TOOLS: tuple[ToolSpec, ...] = (
    _tool(
        ToolName.GET_RUN_STATUS,
        "Whether this run's output is synthetic, verified real, or unverified, and its status, "
        "pipeline profile, and data origin.",
        {OUTPUT_NATURE_CLAIM_ID, "run_status"},
    ),
    _tool(
        ToolName.GET_TRACKING_DIAGNOSTICS,
        "Descriptive tracking diagnostics: detection and tracklet counts, tracklet durations, "
        "assignment, active tracklets, and what the diagnostics do not establish.",
        {
            "diag_tracking",
            "diag_legacy_tracking",
            "diag_synthetic",
            "diag_assignment",
            "diag_duration",
            "diag_active",
            "diag_short_tracklets",
            "diag_insufficient",
            "diag_interpretation",
            "diag_ground_truth",
            "diag_reliability_scope",
        },
    ),
    _tool(
        ToolName.GET_ARTIFACT_METADATA,
        "How many artifacts this run registered, and how many passed integrity verification.",
        {"artifact_integrity"},
    ),
    _tool(
        ToolName.GET_MODEL_PROVENANCE,
        "Recorded model references, code revision, source-video checksum, and source duration.",
        {"provenance", "model_references", "model_details", "source_duration"},
    ),
    _tool(
        ToolName.GET_EVIDENCE_LIMITATIONS,
        "What evidence is missing for this run, and what is never measured or claimed.",
        {"missing_evidence", "no_missing_evidence", "boundaries"},
    ),
)

_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "required": [],
    "additionalProperties": False,
}
_RUN_REFERENCE_RE = re.compile(
    r"run|analysis|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)


def tool_definitions(specs: tuple[ToolSpec, ...] = READ_ONLY_TOOLS) -> tuple[dict[str, Any], ...]:
    """Strict Messages API definitions. ``strict`` keeps a model to the empty schema."""
    return tuple(
        {"name": spec.name.value, "description": spec.description, "strict": True, "input_schema": _INPUT_SCHEMA}
        for spec in specs
    )


def references_a_run(arguments: Any) -> bool:
    """Whether tool arguments try to name a run. Valid input has no arguments at all."""
    if arguments is None or arguments == {}:
        return False
    try:
        text = json.dumps(arguments, default=str)
    except (TypeError, ValueError, RecursionError):
        return False  # still rejected: such a value cannot satisfy the input schema
    return _RUN_REFERENCE_RE.search(text) is not None


@dataclass(frozen=True, slots=True)
class ToolExecution:
    """One finished call. ``content`` is exactly what goes back to the model."""

    tool: ToolName | None
    status: ToolCallStatus
    content: str
    claims: tuple[EvidenceClaim, ...] = ()
    error_code: ToolErrorCode | None = None

    @property
    def is_error(self) -> bool:
        return self.error_code is not None


class ToolRegistry:
    """The tools of one request, bound by the server to that request's claims."""

    def __init__(
        self,
        evidence: ValidatedRunEvidence,
        catalog: tuple[EvidenceClaim, ...],
        specs: tuple[ToolSpec, ...] = READ_ONLY_TOOLS,
    ) -> None:
        if {spec.name for spec in specs} != set(ToolName) or len(specs) != len(ToolName):
            raise ValueError("the agent registry is exactly the approved read-only tools")
        self._output_nature = evidence.output_nature
        self._catalog = catalog
        self._by_id = {claim.claim_id: claim for claim in catalog}
        self._specs: Mapping[str, ToolSpec] = {spec.name.value: spec for spec in specs}
        self._called: set[ToolName] = set()

    def execute(self, name: str, arguments: Any) -> ToolExecution:
        spec = self._specs.get(name)
        if spec is None:
            return _rejected(None, ToolErrorCode.UNKNOWN_TOOL)
        try:
            spec.input_model.model_validate(arguments)
        except ValidationError:  # its message quotes the model's input
            return _rejected(spec.name, ToolErrorCode.INVALID_ARGUMENTS)
        if spec.name in self._called:
            return _rejected(spec.name, ToolErrorCode.DUPLICATE_CALL)
        self._called.add(spec.name)

        try:
            claims = spec.handler(self._catalog)
        except Exception:  # never shown to the model, which only learns the fixed code
            return _rejected(spec.name, ToolErrorCode.TOOL_FAILED, ToolCallStatus.FAILED)
        if not self._approved(spec, claims):
            return _rejected(spec.name, ToolErrorCode.INVALID_TOOL_OUTPUT, ToolCallStatus.FAILED)
        status = ToolCallStatus.OK if claims else ToolCallStatus.NO_EVIDENCE
        try:
            payload = ToolResultPayload(
                tool=spec.name,
                status=status,
                output_nature=self._output_nature,
                claims=[
                    ToolClaim(id=claim.claim_id, text=claim.text, citations=list(claim.citations))
                    for claim in claims
                ],
            )
        except ValidationError:  # a claim outside the payload's bounds is not sent, or truncated
            return _rejected(spec.name, ToolErrorCode.INVALID_TOOL_OUTPUT, ToolCallStatus.FAILED)
        return ToolExecution(spec.name, status, payload.model_dump_json(exclude_none=True), claims)

    def _approved(self, spec: ToolSpec, claims: Any) -> bool:
        """Only this tool's claims, exactly as the server built them for this request."""
        return isinstance(claims, tuple) and all(
            isinstance(claim, EvidenceClaim)
            and claim.claim_id in spec.claims
            and self._by_id.get(claim.claim_id) == claim
            for claim in claims
        )


def _rejected(
    tool: ToolName | None, code: ToolErrorCode, status: ToolCallStatus = ToolCallStatus.REJECTED
) -> ToolExecution:
    return ToolExecution(tool, status, json.dumps({"error": code.value}), error_code=code)


__all__ = [
    "NoInput",
    "READ_ONLY_TOOLS",
    "ToolCallStatus",
    "ToolErrorCode",
    "ToolExecution",
    "ToolName",
    "ToolRegistry",
    "ToolResultPayload",
    "ToolSpec",
    "references_a_run",
    "tool_definitions",
]
