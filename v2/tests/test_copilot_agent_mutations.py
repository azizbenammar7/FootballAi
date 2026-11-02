"""Mutation tests: break each safety property and require a test to notice.

A green suite proved little for this copilot before, so each property below is
deliberately removed from a private copy of the source, and the agent's tests must
then fail. The source tree is never touched, and an unmodified copy must pass first.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path("v2/src/footballai_v2/copilot")
LOOP = PACKAGE / "agent/loop.py"
TOOLS = PACKAGE / "agent/tools.py"
TRACE = PACKAGE / "agent/trace.py"
PROVIDER = PACKAGE / "provider.py"
TARGETS = ["v2/tests/test_copilot_agent.py", "v2/tests/test_copilot_agent_model.py"]

MUTANTS = {
    "router_no_longer_stops_unsupported_questions": (
        LOOP,
        "if route.scope is QuestionScope.UNSUPPORTED:",
        "if False:",
    ),
    "missing_evidence_reaches_the_model": (LOOP, "if not can_answer(catalog, route):", "if False:"),
    "tools_see_claims_the_question_cannot_use": (
        LOOP,
        "catalog = eligible_claims(build_claim_catalog(evidence), route)",
        "catalog = build_claim_catalog(evidence)",
    ),
    "unretrieved_selection_is_accepted": (
        LOOP,
        "known = tuple(claim_id for claim_id in claim_ids if claim_id in retrieved)",
        "known = claim_ids",
    ),
    "the_trace_lists_what_was_selected_not_shown": (
        LOOP,
        "rendered=tuple(claim.claim_id for claim in shown),",
        "rendered=(OUTPUT_NATURE_CLAIM_ID, *known),",
    ),
    "citations_are_not_validated": (LOOP, "validate_citations(rendered[1], evidence, run_id)", "pass"),
    "citations_checked_against_the_evidence_run_not_the_requested_run": (
        LOOP,
        "validate_citations(rendered[1], evidence, run_id)",
        "validate_citations(rendered[1], evidence, evidence.run_id)",
    ),
    "cross_run_arguments_do_not_stop_the_request": (
        LOOP,
        "if any(references_a_run(call.arguments) for call in calls):",
        "if False:",
    ),
    "tool_budget_is_ignored": (
        LOOP,
        "elif recorder.tool_call_count + len(calls) > self._limits.max_tool_calls:",
        "elif False:",
    ),
    "tool_budget_off_by_one": (
        LOOP,
        "elif recorder.tool_call_count + len(calls) > self._limits.max_tool_calls:",
        "elif recorder.tool_call_count + len(calls) > self._limits.max_tool_calls + 1:",
    ),
    "the_last_turn_still_runs_tools": (LOOP, "elif final_turn:", "elif False:"),
    "deadline_is_ignored": (LOOP, "if remaining <= 0:", "if False:"),
    "a_turn_is_not_cut_to_the_remaining_time": (
        LOOP,
        "timeout = min(limits.model_timeout_seconds, remaining)",
        "timeout = limits.model_timeout_seconds",
    ),
    "late_replies_are_kept": (LOOP, "if elapsed > request.timeout_seconds:", "if False:"),
    "prompt_budget_is_ignored": (LOOP, "if prompt_chars > self._settings.max_prompt_chars:", "if False:"),
    "rejected_calls_do_not_set_is_error": (LOOP, "if execution.is_error:", "if False:"),
    "assistant_content_is_rebuilt": (
        LOOP,
        '{"role": "assistant", "content": outcome.assistant_content}',
        '{"role": "assistant", "content": list(outcome.assistant_content)[1:]}',
    ),
    "limits_can_be_raised": (LOOP, "if not 0 <= self.max_tool_calls <= MAX_TRACED_TOOL_CALLS:", "if False:"),
    "unknown_tools_get_a_fallback": (TOOLS, "spec = self._specs.get(name)", "spec = self._specs.get(name, READ_ONLY_TOOLS[0])"),
    "tool_arguments_are_not_validated": (TOOLS, "spec.input_model.model_validate(arguments)", "pass"),
    "duplicate_calls_run_again": (TOOLS, "if spec.name in self._called:", "if False:"),
    "an_oversized_claim_raises": (TOOLS, "        except ValidationError:  # a claim outside", "        except KeyError:  # a claim outside"),
    "tool_output_is_trusted": (TOOLS, "if not self._approved(spec, claims):", "if False:"),
    "a_tool_may_return_another_tools_claim": (TOOLS, "and claim.claim_id in spec.claims\n", "\n"),
    "tool_exception_text_reaches_the_model": (
        TOOLS,
        "except Exception:  # never shown to the model, which only learns the fixed code\n"
        "            return _rejected(spec.name, ToolErrorCode.TOOL_FAILED, ToolCallStatus.FAILED)",
        "except Exception as exc:\n"
        "            return ToolExecution(spec.name, ToolCallStatus.FAILED, str(exc), error_code=ToolErrorCode.TOOL_FAILED)",
    ),
    "tools_are_not_strict": (TOOLS, '"strict": True', '"strict": False'),
    "a_tool_accepts_a_run_id": (TOOLS, '    "properties": {},', '    "properties": {"run_id": {"type": "string"}},'),
    "registry_accepts_any_tools": (
        TOOLS,
        "if {spec.name for spec in specs} != set(ToolName) or len(specs) != len(ToolName):",
        "if False:",
    ),
    "model_stop_reason_is_copied_into_the_trace": (
        TRACE,
        "if isinstance(stop_reason, str) and stop_reason in _KNOWN_STOP_REASONS",
        "if isinstance(stop_reason, str)",
    ),
    "provider_request_id_is_copied_into_the_trace": (
        TRACE,
        "request_id if isinstance(request_id, str) and _REQUEST_ID_RE.fullmatch(request_id) else None",
        "request_id if isinstance(request_id, str) else None",
    ),
    "tool_use_on_a_final_turn_is_ignored": (PROVIDER, "    if blocks:\n        raise _invalid_reply(telemetry)\n", ""),
    "a_tool_stop_without_a_call_is_a_final_turn": (
        PROVIDER,
        '    if stop_reason == "tool_use":\n        if not blocks:\n            raise _invalid_reply(telemetry)\n',
        '    if stop_reason == "tool_use" and blocks:\n',
    ),
    "unknown_stop_reasons_continue": (
        PROVIDER,
        'if stop_reason not in ("tool_use", "end_turn"):',
        "if False:",
    ),
    "repeated_tool_use_ids_are_accepted": (
        PROVIDER,
        "if not valid or len({call.call_id for call in calls}) != len(calls):",
        "if not valid:",
    ),
    "the_adapter_uses_sdk_retries": (
        PROVIDER,
        "anthropic.Anthropic(api_key=api_key, max_retries=0)",
        "anthropic.Anthropic(api_key=api_key)",
    ),
    "the_adapter_ignores_the_opt_in": (
        PROVIDER,
        "if not self._settings.wants_real_provider or not api_key:",
        "if not api_key:",
    ),
}


@pytest.fixture(scope="module")
def mirror(tmp_path_factory):
    root = tmp_path_factory.mktemp("mutants")
    ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache")
    shutil.copytree(REPO_ROOT / "v2/src", root / "v2/src", ignore=ignore)
    shutil.copytree(REPO_ROOT / "evaluation/copilot", root / "evaluation/copilot", ignore=ignore)
    (root / "v2/tests").mkdir()
    for path in (REPO_ROOT / "v2/tests").glob("*.py"):
        if path.name == "conftest.py" or path.name.startswith("test_copilot_"):
            shutil.copy(path, root / "v2/tests" / path.name)
    return root


def run_tests(root: Path) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PYTHONPATH": str(root / "v2/src"),
        "PYTHONDONTWRITEBYTECODE": "1",  # a stale .pyc must not hide a mutant
    }
    env.pop("ANTHROPIC_API_KEY", None)
    command = [sys.executable, "-m", "pytest", *TARGETS, "-x", "-q", "-p", "no:cacheprovider", "--no-header"]
    return subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)


def test_an_unmodified_copy_passes(mirror):
    result = run_tests(mirror)
    assert result.returncode == 0, result.stdout[-2000:]


@pytest.mark.parametrize("name", MUTANTS)
def test_mutant_is_caught(mirror, name):
    relative, original, mutated = MUTANTS[name]
    path = mirror / relative
    source = path.read_text(encoding="utf-8")
    assert source.count(original) >= 1, f"{name} no longer matches the source"
    path.write_text(source.replace(original, mutated), encoding="utf-8")
    try:
        result = run_tests(mirror)
    finally:
        path.write_text(source, encoding="utf-8")
    assert result.returncode == 1, f"{name} survived (exit {result.returncode}):\n{result.stdout[-1500:]}"
