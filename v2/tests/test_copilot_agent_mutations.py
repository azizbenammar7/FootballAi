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
CONFIG = PACKAGE / "config.py"
BUDGET = PACKAGE / "budget.py"
REAL_AGENT = PACKAGE / "real_agent.py"
GROUNDING = PACKAGE / "grounding.py"
APP = Path("v2/src/footballai_v2/api/app.py")
ROUTE = Path("v2/src/footballai_v2/api/agent_route.py")
TARGETS = ["v2/tests/test_copilot_agent.py", "v2/tests/test_copilot_agent_model.py"]
API_TARGETS = ["v2/tests/test_copilot_agent_api.py"]
REAL_TARGETS = ["v2/tests/test_copilot_agent_real_gate.py"]
PROTOCOL_TARGETS = ["v2/tests/test_copilot_agent_sdk_protocol.py"]

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
        "anthropic.Anthropic(api_key=api_key, base_url=base_url, max_retries=0)",
        "anthropic.Anthropic(api_key=api_key, base_url=base_url)",
    ),
    "the_adapter_ignores_the_opt_in": (
        PROVIDER,
        "if not self._settings.wants_real_provider or not api_key or base_url is None:",
        "if not api_key or base_url is None:",
    ),
}

# The real path's gates. These need no SDK: an adapter that counts calls stands in.
REAL_MUTANTS = {
    "an_unapproved_endpoint_keeps_the_key": (
        PROVIDER,
        "if (parts.scheme == \"https\" and host == \"api.anthropic.com\") or (parts.scheme == \"http\" and host in _LOOPBACK_HOSTS):",
        "if True:",
    ),
    "a_hosted_platform_is_not_refused": (
        REAL_AGENT,
        '"hosted_environment": any(environment.get(name) for name in _HOSTED_MARKERS),',
        '"hosted_environment": False,',
    ),
    "continuous_integration_is_not_refused": (
        REAL_AGENT,
        '"continuous_integration": any(environment.get(name) for name in _CI_MARKERS),',
        '"continuous_integration": False,',
    ),
    "the_real_path_needs_no_cost_cap": (REAL_AGENT, '"no_cost_cap": settings.agent_max_cost_usd <= 0,', '"no_cost_cap": False,'),
    "the_real_path_needs_no_request_cap": (REAL_AGENT, '"no_request_cap": settings.agent_max_requests < 1,', '"no_request_cap": False,'),
    "an_unpriced_model_is_allowed": (REAL_AGENT, '"unpriced_model": settings.model_name not in MODEL_PRICES,', '"unpriced_model": False,'),
    "a_blocked_real_path_builds_anyway": (REAL_AGENT, "    if blockers:\n        raise RealAgentRefused(blockers)\n", ""),
    "the_real_agent_opens_explain_as_a_billed_route": (
        PROVIDER,
        "if settings.wants_real_provider and not settings.agent_real_model_enabled:",
        "if settings.wants_real_provider:",
    ),
    "the_budget_is_not_checked_before_sending": (BUDGET, "            self._guard.admit(worst_case)\n", "            pass\n"),
    "the_money_cap_is_ignored": (
        BUDGET,
        "if self.spent_usd + self.reserved_usd + worst_case_usd > self.max_cost_usd:",
        "if False:",
    ),
    "in_flight_requests_reserve_nothing": (BUDGET, "self.spent_usd + self.reserved_usd + worst_case_usd", "self.spent_usd + worst_case_usd"),
    "a_failed_request_is_not_charged": (BUDGET, "            self._guard.charge(exc.telemetry, worst_case)\n", ""),
    "a_timeout_is_free": (
        BUDGET,
        "                self.requests_at_worst_case += 1\n                self.spent_usd += worst_case_usd\n",
        "                self.requests_at_worst_case += 1\n",
    ),
    "earlier_turns_add_nothing_to_the_bound": (BUDGET, "prompt = _server_bytes(request) + earlier_turns * request.max_output_tokens", "prompt = _server_bytes(request)"),
    "a_budget_refusal_counts_as_a_turn": (LOOP, "            if exc.kind not in _NEVER_SENT:\n                recorder.record_turn", "            if True:\n                recorder.record_turn"),
    "stub_tokens_look_measured": (TRACE, "        if mode is ModelMode.OFFLINE_DEMO_FAKE:\n            return UsageSource.SCRIPTED_CONSTANT\n", ""),
    "the_estimate_never_admits_a_gap": (TRACE, "cost_is_lower_bound=estimated is not None and not self._usage_complete,", "cost_is_lower_bound=False,"),
    "the_route_falls_back_to_the_script": (
        ROUTE,
        "    if settings.agent_real_model_enabled:\n        from footballai_v2.copilot.real_agent import real_agent_factory\n\n        return real_agent_factory(settings)\n",
        "",
    ),
    "the_adapter_route_skips_the_local_guard": (
        ROUTE,
        '''        response_model=AgentAdapterInfo,
        dependencies=[Depends(local_callers_only)],
''',
        "        response_model=AgentAdapterInfo,\n",
    ),
    "the_adapter_route_always_says_scripted": (
        ROUTE,
        "return AgentAdapterInfo(adapter=AgentAdapter.for_mode(model.mode), model=model.model_name)",
        "return AgentAdapterInfo(adapter=AgentAdapter.OFFLINE_SCRIPTED, model=model.model_name)",
    ),
    "the_model_is_never_closed": (ROUTE, "            if close is not None:\n                close()\n", ""),
}

# Needs the SDK and a dripping server, so it runs only where the SDK is installed.
ABORT_MUTANTS = {
    "a_slow_call_is_only_timed_by_the_sdk": (PROVIDER, "worker.join(request.timeout_seconds)", "worker.join()"),
    "the_abort_leaves_the_connection_open": (
        PROVIDER,
        "        if worker.is_alive():\n            self.close()\n",
        "        if worker.is_alive():\n",
    ),
    "a_refusal_category_is_copied_verbatim": (
        PROVIDER,
        'return category if category in _REFUSAL_CATEGORIES else "other"',
        "return category",
    ),
}

# The route's boundaries, broken one at a time. These need the HTTP tests.
API_MUTANTS = {
    "the_agent_route_ignores_its_own_flag": (APP, "if gate.agent_enabled:", "if True:"),
    "the_agent_flag_opens_without_the_copilot_gate": (
        CONFIG,
        'agent_enabled = enabled and _flag("FOOTBALLAI_COPILOT_AGENT_ENABLED")',
        'agent_enabled = _flag("FOOTBALLAI_COPILOT_AGENT_ENABLED")',
    ),
    "a_remote_peer_reaches_the_agent": (ROUTE, "peer in _LOOPBACK_PEERS\n", "True\n"),
    "a_tunnel_or_rebound_host_reaches_the_agent": (ROUTE, "and request.url.hostname in _LOOPBACK_HOSTS\n", "\n"),
    "another_local_page_reaches_the_agent": (ROUTE, "and (origin is None or origin in allowed_origins)\n", "\n"),
    "provider_failure_classes_are_published": (
        TRACE,
        "tool_error_code=trace.error_code if isinstance(trace.error_code, ToolErrorCode) else None,",
        "tool_error_code=None if trace.error_code is None else ToolErrorCode.TOOL_FAILED,",
    ),
    "one_adapter_is_shared_between_requests": (
        ROUTE,
        "build_model = model_factory or _default_model_factory(settings)",
        "shared = (model_factory or _default_model_factory(settings))(); build_model = lambda: shared",
    ),
    "an_unfinished_run_reaches_the_agent": (
        ROUTE,
        "        require_completed(run_id)\n",
        "",
    ),
    "the_trace_loses_its_correlation_with_the_response": (
        ROUTE,
        "correlation_id=request.state.request_id",
        "correlation_id=None",
    ),
    "a_real_setting_builds_a_real_adapter": (
        ROUTE,
        "build_model = model_factory or _default_model_factory(settings)",
        "from footballai_v2.copilot.provider import AnthropicAgentModel\n"
        "    build_model = model_factory or (\n"
        "        (lambda: AnthropicAgentModel(settings))\n"
        "        if settings.wants_real_provider\n"
        "        else _default_model_factory(settings)\n"
        "    )",
    ),
    "the_no_gap_claim_is_offered_beside_a_gap": (GROUNDING, "elif _nothing_is_missing(evidence):", "if True:"),
    "the_no_gap_claim_ignores_a_failed_artifact": (
        GROUNDING,
        'and all(item.integrity_state == "verified" for item in evidence.artifacts)\n',
        "\n",
    ),
    "the_no_gap_claim_ignores_an_unverified_run": (
        GROUNDING,
        "and evidence.output_nature is not OutputNature.UNVERIFIED_OR_PARTIAL\n",
        "\n",
    ),
    "the_no_gap_claim_ignores_a_refused_measure": (
        GROUNDING,
        "        and not _withheld_a_measure(evidence)\n",
        "\n",
    ),
    "an_impossible_fraction_is_not_a_refused_measure": (
        GROUNDING,
        "return (fraction is not None and fraction > 1) or len(",
        "return len(",
    ),
    "the_no_gap_claim_loses_its_artifact_citation": (
        GROUNDING,
        '                diagnostic_paths=("diagnostics.schema",),\n            )\n        )\n    claims.append(',
        '            )\n        )\n    claims.append(',
    ),
    "the_no_gap_claim_joins_the_summary": (
        GROUNDING,
        '_LIMIT_CLAIMS = frozenset({"missing_evidence", "boundaries"})',
        '_LIMIT_CLAIMS = frozenset({"missing_evidence", "no_missing_evidence", "boundaries"})',
    ),
}


SCORING = Path("v2/src/footballai_v2/copilot/evaluation/agent_scoring.py")
RULE = Path("v2/src/footballai_v2/copilot/evaluation/agent_rule.py")
LIVE = Path("v2/src/footballai_v2/copilot/evaluation/agent_live.py")
SCORING_TARGETS = ["v2/tests/test_copilot_agent_scoring_v2.py"]

# M3-B0.1. Each mutant names the test that must catch it, so a mutant cannot pass by
# breaking something else: (file, original, mutated, the test that must fail).
SCORING_MUTANTS = {
    "an_off_topic_claim_is_still_relevant": (
        SCORING,
        "relevant = bool(answering_claims & shown) and content <= permitted",
        "relevant = bool(answering_claims & shown)",
        "test_the_right_claim_padded_with_an_off_topic_one_is_irrelevant",
    ),
    "the_always_shown_label_counts_as_an_answer": (
        SCORING,
        "answering_claims = required - {OUTPUT_NATURE_CLAIM_ID} or required",
        "answering_claims = required",
        "test_the_always_shown_label_does_not_make_an_answer_relevant",
    ),
    "completeness_is_judged_for_a_false_answer": (
        SCORING,
        "        if expecting:  # completeness is only judged where an answer was wanted\n            complete = required <= shown",
        "        complete = required <= shown",
        "test_completeness_is_not_judged_for_a_question_with_no_answer",
    ),
    "a_turnless_stop_leaves_the_tool_denominator": (
        SCORING,
        "        tools_correct = False  # stopped before its first turn: nothing right was run",
        "        pass",
        "test_a_question_stopped_before_its_first_turn_counts_as_wrong_tool_selection",
    ),
    "latency_includes_operator_stops": (
        SCORING,
        "return self.attempted and self.owner is not Owner.OPERATOR",
        "return self.attempted",
        "test_latency_leaves_out_operator_stops",
    ),
    "untracked_code_is_ignored": (
        LIVE,
        "return head or None, not changed and not untracked_code",
        "return head or None, not changed",
        "test_an_untracked_module_under_the_source_tree_is_a_modified_tree",
    ),
    "a_partial_answer_is_a_success": (
        SCORING,
        "task_success = bool(answered and relevant and complete and v1.citations_valid) and safe",
        "task_success = bool(answered and relevant and v1.citations_valid) and safe",
        "test_a_relevant_but_partial_answer_is_incomplete",
    ),
    "an_answerable_abstention_is_not_judged": (
        SCORING,
        "abstention_correct = False if termination is TerminationClass.ABSTAINED else None",
        "abstention_correct = None",
        "test_abstaining_on_an_answerable_question_is_an_incorrect_abstention",
    ),
    "a_tool_budget_stop_is_nobodys_failure": (
        SCORING,
        "AgentStopReason.TOOL_BUDGET_EXHAUSTED: (TerminationClass.TOOL_BUDGET_EXHAUSTED, Owner.AGENT)",
        "AgentStopReason.TOOL_BUDGET_EXHAUSTED: (TerminationClass.TOOL_BUDGET_EXHAUSTED, Owner.NONE)",
        "test_budget_turn_and_deadline_stops_count_as_agent_non_completion",
    ),
    "a_deadline_is_blamed_on_the_provider": (
        SCORING,
        "AgentStopReason.DEADLINE_EXCEEDED: (TerminationClass.DEADLINE_EXCEEDED, Owner.AGENT)",
        "AgentStopReason.DEADLINE_EXCEEDED: (TerminationClass.DEADLINE_EXCEEDED, Owner.PROVIDER)",
        "test_each_injected_fault_is_classified_once_and_owned",
    ),
    "a_refusal_is_a_provider_failure": (
        SCORING,
        "AgentStopReason.MODEL_REFUSAL: (TerminationClass.MODEL_REFUSAL, Owner.AGENT)",
        "AgentStopReason.MODEL_REFUSAL: (TerminationClass.MODEL_REFUSAL, Owner.PROVIDER)",
        "test_each_injected_fault_is_classified_once_and_owned",
    ),
    "operator_stops_join_the_denominator": (
        SCORING,
        "return self.attempted and self.owner is not Owner.OPERATOR",
        "return self.attempted",
        "test_no_termination_disappears_and_none_is_counted_twice",
    ),
    "invalid_arguments_are_not_counted": (
        SCORING,
        "call.error_code is ToolErrorCode.INVALID_ARGUMENTS for call in trace.tool_calls",
        "False for call in trace.tool_calls",
        "test_invalid_tool_arguments_are_an_event_not_a_termination",
    ),
    "new_false_answers_count_every_paired_case": (
        SCORING,
        "            rule_right_non_answers,\n        ),\n        \"live_tool_selection_accuracy\"",
        "            paired,\n        ),\n        \"live_tool_selection_accuracy\"",
        "test_finding_b_new_false_answers_has_no_opportunities_on_heldout",
    ),
    "replay_counts_toward_live_citations": (
        SCORING,
        '"live_citation_validity": _rate(score.citations_valid for score in live)',
        '"live_citation_validity": _rate(score.citations_valid for score in every)',
        "test_finding_e_replay_no_longer_counts_toward_citation_validity",
    ),
    "no_opportunities_reads_as_a_pass": (
        RULE,
        "    if n == 0:\n        return result(\"not_evaluable\", \"no_opportunities\")",
        "    if False:\n        return result(\"not_evaluable\", \"no_opportunities\")",
        "test_every_criterion_status_carries_a_reason",
    ),
    "not_evaluable_reads_as_a_thin_sample": (
        RULE,
        'if statuses & {"not_evaluable", "not_measured"}:',
        'if statuses & {"not_measured"}:',
        "test_not_evaluable_outranks_a_thin_sample",
    ),
    "the_approval_ignores_the_commit": (
        LIVE,
        '"approval_other_commit": bool(provenance.get("source_commit"))\n        and approval.source_commit == provenance.get("source_commit"),',
        '"approval_other_commit": True,',
        "test_each_pin_is_enforced",
    ),
    "the_approval_ignores_a_modified_tree": (
        LIVE,
        '"source_tree_modified": provenance.get("source_tree_clean") is True,',
        '"source_tree_modified": True,',
        "test_each_pin_is_enforced",
    ),
    "the_approval_ignores_the_scorer": (
        LIVE,
        '"approval_other_scorer": approval.scorer_version == provenance.get("scorer_version"),',
        '"approval_other_scorer": True,',
        "test_each_pin_is_enforced",
    ),
    "a_probe_has_no_cap": (
        LIVE,
        "or approval.max_model_requests <= PROTOCOL_PROBE_MAX_REQUESTS,",
        "or True,",
        "test_each_stage_has_its_own_bounds",
    ),
}
BENCHMARK = Path("v2/src/footballai_v2/copilot/evaluation/agent_benchmark.py")
BENCHMARK_CLI = Path("v2/src/footballai_v2/cli/copilot_agent_benchmark.py")
DATASET_TARGETS = ["v2/tests/test_copilot_agent_dataset_v3.py"]

# M3-B0.2. The v3 leakage checks, the operator-stop fix, and the unfrozen-dataset refusal.
DATASET_MUTANTS = {
    "exact_leakage_is_not_checked": (
        BENCHMARK,
        "        if plain in exact:\n",
        "        if False:\n",
        "test_the_v3_loader_refuses_held_out_leakage",
    ),
    "near_duplicates_are_not_checked": (
        BENCHMARK,
        "            if similar is not None:\n",
        "            if False:\n",
        "test_the_v3_loader_refuses_held_out_leakage",
    ),
    "prompt_text_is_not_checked": (
        BENCHMARK,
        'if plain and f" {plain} " in text',
        "if False",
        "test_the_v3_loader_refuses_held_out_leakage",
    ),
    "v3_loads_without_its_leakage_check": (
        BENCHMARK,
        "    if dev.dataset_version == DATASET_VERSION_V3:\n        leaks",
        "    if False:\n        leaks",
        "test_the_v3_loader_refuses_held_out_leakage",
    ),
    "a_v3_label_may_omit_its_origin": (
        BENCHMARK,
        "if self.dataset_version == DATASET_VERSION_V3 and False in has_origin:",
        "if False:",
        "test_every_v3_label_states_its_origin_and_v2_none",
    ),
    "operator_stops_count_against_quality": (
        SCORING,
        "return self.attempted and self.owner is not Owner.OPERATOR",
        "return self.attempted",
        "test_operator_stops_leave_every_quality_denominator",
    ),
    "the_signature_vocabulary_drifts_from_routing": (
        BENCHMARK,
        '    "a about all an analysis',
        '    "a about all an analyses',
        "test_the_pinned_signature_vocabulary_matches_routing_v2",
    ),
    "a_spend_cap_reads_as_a_finished_run": (
        RULE,
        '"verdict_if_accepted": _overall(as_bound, results, completed=completed).value,',
        '"verdict_if_accepted": _overall(as_bound, results, completed=True).value,',
        "test_a_spend_cap_is_an_incomplete_run_never_a_model_failure",
    ),
    "an_unfrozen_dataset_is_scored": (
        BENCHMARK_CLI,
        "    if not all((args.dataset / name).is_file() for name in (RULE_FILE, RULE_FILE_V2)):",
        "    if False:",
        "test_no_paid_or_replay_run_starts_on_a_dataset_without_its_own_rule",
    ),
}
# A mutant that "fails" this way broke the module, not the property under test.
WRONG_REASONS = ("SyntaxError", "IndentationError", "ImportError", "ModuleNotFoundError", "NameError")


@pytest.fixture(scope="module")
def mirror(tmp_path_factory):
    root = tmp_path_factory.mktemp("mutants")
    ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache")
    shutil.copytree(REPO_ROOT / "v2/src", root / "v2/src", ignore=ignore)
    shutil.copytree(REPO_ROOT / "evaluation/copilot", root / "evaluation/copilot", ignore=ignore)
    (root / "docker").mkdir()
    for name in ("api.Dockerfile", "worker.Dockerfile"):
        shutil.copy(REPO_ROOT / "docker" / name, root / "docker" / name)
    workloads = Path("infra/terraform/environments/staging/workloads.tf")
    (root / workloads.parent).mkdir(parents=True)
    shutil.copy(REPO_ROOT / workloads, root / workloads)
    shutil.copytree(REPO_ROOT / "v2/dashboard/src", root / "v2/dashboard/src", ignore=ignore)
    (root / "v2/tests").mkdir()
    for path in (REPO_ROOT / "v2/tests").glob("*.py"):
        if path.name == "conftest.py" or path.name.startswith("test_copilot_"):
            shutil.copy(path, root / "v2/tests" / path.name)
    return root


def run_tests(root: Path, targets: list[str] = TARGETS, *, stop_at_first: bool = True) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PYTHONPATH": str(root / "v2/src"),
        "PYTHONDONTWRITEBYTECODE": "1",  # a stale .pyc must not hide a mutant
    }
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "CI", "GITHUB_ACTIONS"):
        env.pop(name, None)
    command = [sys.executable, "-m", "pytest", *targets, *(["-x"] if stop_at_first else []), "-q", "-p", "no:cacheprovider", "--no-header", "-rf"]
    return subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)


def assert_caught(root: Path, name: str, mutant: tuple[Path, str, str], targets: list[str]) -> None:
    relative, original, mutated = mutant
    path = root / relative
    source = path.read_text(encoding="utf-8")
    assert source.count(original) >= 1, f"{name} no longer matches the source"
    path.write_text(source.replace(original, mutated), encoding="utf-8")
    try:
        result = run_tests(root, targets)
    finally:
        path.write_text(source, encoding="utf-8")
    assert result.returncode == 1, f"{name} survived (exit {result.returncode}):\n{result.stdout[-1500:]}"
    wrong = [marker for marker in WRONG_REASONS if marker in result.stdout]
    assert not wrong, f"{name} failed for the wrong reason ({wrong}):\n{result.stdout[-1500:]}"


def assert_caught_by(root: Path, name: str, mutant: tuple[Path, str, str, str], targets: list[str]) -> None:
    """Without ``-x``, so the named test is seen failing even when another fails first."""
    relative, original, mutated, expected_test = mutant
    path = root / relative
    source = path.read_text(encoding="utf-8")
    assert source.count(original) == 1, f"{name} no longer matches the source exactly once"
    path.write_text(source.replace(original, mutated), encoding="utf-8")
    try:
        result = run_tests(root, targets, stop_at_first=False)
    finally:
        path.write_text(source, encoding="utf-8")
    assert result.returncode == 1, f"{name} survived (exit {result.returncode}):\n{result.stdout[-1500:]}"
    wrong = [marker for marker in WRONG_REASONS if marker in result.stdout]
    assert not wrong, f"{name} failed for the wrong reason ({wrong}):\n{result.stdout[-1500:]}"
    failed = [line for line in result.stdout.splitlines() if line.startswith("FAILED ")]
    assert any(f"::{expected_test}" in line for line in failed), f"{name}: {expected_test} did not fail:\n{failed}"


def test_an_unmodified_copy_passes(mirror):
    result = run_tests(mirror, [*TARGETS, *API_TARGETS, *REAL_TARGETS, *SCORING_TARGETS])
    assert result.returncode == 0, result.stdout[-2000:]


@pytest.mark.parametrize("name", SCORING_MUTANTS)
def test_scoring_mutant_is_caught_by_the_test_named_for_it(mirror, name):
    assert_caught_by(mirror, name, SCORING_MUTANTS[name], SCORING_TARGETS)


@pytest.mark.parametrize("name", DATASET_MUTANTS)
def test_dataset_mutant_is_caught_by_the_test_named_for_it(mirror, name):
    assert_caught_by(mirror, name, DATASET_MUTANTS[name], DATASET_TARGETS)


@pytest.mark.parametrize("name", MUTANTS)
def test_mutant_is_caught(mirror, name):
    assert_caught(mirror, name, MUTANTS[name], TARGETS)


@pytest.mark.parametrize("name", API_MUTANTS)
def test_api_mutant_is_caught(mirror, name):
    assert_caught(mirror, name, API_MUTANTS[name], API_TARGETS)


@pytest.mark.parametrize("name", REAL_MUTANTS)
def test_real_path_mutant_is_caught(mirror, name):
    assert_caught(mirror, name, REAL_MUTANTS[name], REAL_TARGETS)


@pytest.mark.parametrize("name", ABORT_MUTANTS)
def test_abort_mutant_is_caught(mirror, name):
    pytest.importorskip("anthropic")
    assert_caught(mirror, name, ABORT_MUTANTS[name], PROTOCOL_TARGETS)
