"""The agent evaluation: reproducible, offline, and able to fail."""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from footballai_v2.cli import copilot_agent_eval
from footballai_v2.copilot.agent import AgentStopReason, ToolName
from footballai_v2.copilot.evaluation.agent_dataset import AgentCase, AgentExpected, ScriptStep, load_agent_dataset
from footballai_v2.copilot.evaluation.agent_harness import run_scripted_case, tool_metrics
from footballai_v2.copilot.evaluation.cost import MODEL_PRICES, agent_question_worst_case_usd, worst_case_cost_usd
from footballai_v2.copilot.evaluation.dataset import Suite
from footballai_v2.copilot.models import OutputNature

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "evaluation" / "copilot" / "agent-v1"
AGENT_SOURCE = REPO_ROOT / "v2" / "src" / "footballai_v2" / "copilot" / "agent"


@pytest.fixture(scope="module")
def dataset():
    return load_agent_dataset(DATASET_DIR)


@pytest.fixture(scope="module")
def report(dataset):
    return copilot_agent_eval.build_report(dataset)


def test_the_committed_report_is_exactly_what_the_code_produces(report):
    committed = (DATASET_DIR / "baseline-report.json").read_text(encoding="utf-8")
    assert committed == json.dumps(report, indent=2) + "\n"


def test_the_report_states_that_no_model_was_called(report):
    assert report["mode"] == "dry_run"
    assert report["live_model_requests"] == 0
    assert "No language model was called" in report["note"]


def test_every_scripted_case_meets_its_expectations(report):
    suite = report["scripted_suite"]
    assert suite["failed_cases"] == []
    assert suite["all_expectations_met"]["correct"] == suite["cases"] == 69
    assert suite["protocol_violations"] == 0


def test_no_hostile_script_produced_an_unlabelled_answer(report):
    suite = report["scripted_suite"]
    assert suite["answers_not_labelled_as_answers"] == []
    assert suite["unsupported_claim_rate"]["unpermitted"] == 0
    assert suite["citation_validity"]["rate"] == 1.0


def test_every_stop_reason_the_loop_can_give_is_exercised(report):
    unreachable_here = {
        AgentStopReason.CITATION_REJECTED,  # needs a tampered catalog: test_copilot_agent.py
        AgentStopReason.TURN_BUDGET_EXHAUSTED,  # needs tighter limits: test_copilot_agent.py
        AgentStopReason.PROMPT_BUDGET_EXCEEDED,  # needs a tighter budget: test_copilot_agent.py
    }
    assert set(report["scripted_suite"]["stops"]) == {reason.value for reason in AgentStopReason} - {
        reason.value for reason in unreachable_here
    }


def test_the_agent_path_loses_nothing_to_the_deterministic_route(report):
    comparison = report["deterministic_comparison"]
    assert comparison["paired"]["paired_cases"] == 88
    assert comparison["paired"]["improved"] == comparison["paired"]["regressed"] == []
    following = comparison["route_following"]
    assert following["claim_selection_accuracy"] == comparison["deterministic_fake"]["claim_selection_accuracy"]
    assert following["abstention_accuracy"] == comparison["deterministic_fake"]["abstention_accuracy"]
    assert following["fabricated_answers"] == 0


def test_the_one_tool_selection_miss_is_the_known_label_artefact(report):
    accuracy = report["deterministic_comparison"]["route_following"]["tool_selection_accuracy"]
    assert (accuracy["correct"], accuracy["n"]) == (53, 54)


def test_the_metrics_can_fail(report):
    controls = report["negative_controls"]
    for control in controls.values():
        assert control["claim_selection_accuracy"]["rate"] < 0.1
        assert control["tool_selection_accuracy"]["rate"] < 0.1
        assert control["fabricated_answers"] == 0  # bad behaviour stays safe


def test_each_negative_control_fails_the_way_it_should(report):
    controls = report["negative_controls"]
    assert controls["every_tool_at_once"]["stops"]["tool_budget_exhausted"] == 54
    assert controls["skips_retrieval"]["stops"]["selection_rejected"] == 54
    assert controls["always_the_wrong_tool"]["stops"]["no_relevant_claim"] == 40


def test_recorded_invalid_replies_all_fail_closed_through_the_agent(report):
    replay = report["invalid_reply_replay"]
    assert replay["failed_cases"] == []
    assert replay["fail_closed"]["correct"] == replay["fail_closed"]["n"] == 18


def test_the_cost_plan_is_marked_as_a_bound(report):
    plan = report["cost_plan"]
    assert "not a measurement" in plan["note"]
    assert 0 < plan["worst_case_usd_per_question"] < 0.05


def test_the_cost_bound_covers_output_resent_on_later_turns():
    price = MODEL_PRICES["claude-haiku-5-5"]
    bound = agent_question_worst_case_usd(price, 24_576, 1024, 5)
    assert bound > 5 * worst_case_cost_usd(price, 24_576, 1024)
    assert bound == pytest.approx(sum(worst_case_cost_usd(price, 24_576 + turn * 1024, 1024) for turn in range(5)))


def test_a_wrong_expectation_is_detected(dataset):
    case = next(case for case in dataset.cases if case.case_id == "status-one-tool")
    wrong_tool = case.model_copy(
        update={"expected": case.expected.model_copy(update={"tools_executed": (ToolName.GET_MODEL_PROVENANCE,)})}
    )
    assert "tools_executed" in run_scripted_case(wrong_tool, dataset).failures
    wrong_stop = case.model_copy(
        update={"expected": case.expected.model_copy(update={"stop": AgentStopReason.NO_RELEVANT_CLAIM})}
    )
    assert "stop" in run_scripted_case(wrong_stop, dataset).failures


def test_a_shown_claim_outside_the_labels_is_detected(dataset):
    case = next(case for case in dataset.cases if case.case_id == "tracking-one-tool")
    narrow = case.model_copy(
        update={
            "expected": case.expected.model_copy(
                update={"permitted_claims": ("diag_tracking",)}
            )
        }
    )
    assert "claims" in run_scripted_case(narrow, dataset).failures


def test_tool_metrics_separate_useful_and_wasted_calls(dataset):
    by_id = {case.case_id: run_scripted_case(case, dataset).run for case in dataset.cases}
    metrics = tool_metrics([by_id["wrong-tool-then-right-tool"]])
    assert (metrics["tool_calls"], metrics["useful_calls"], metrics["empty_calls"]) == (2, 1, 1)


def test_the_fixtures_are_pinned_by_hash(tmp_path):
    root = tmp_path / "copilot"
    shutil.copytree(REPO_ROOT / "evaluation" / "copilot", root)
    fixtures = root / "v2" / "fixtures.json"
    fixtures.write_text(fixtures.read_text(encoding="utf-8").replace("Valid diagnostics", "Altered"), encoding="utf-8")
    with pytest.raises(ValueError, match="new agent dataset version"):
        load_agent_dataset(root / "agent-v1")


def test_a_script_step_takes_exactly_one_action():
    with pytest.raises(ValueError):
        ScriptStep(call=("get_run_status",), select=("run_status",))
    with pytest.raises(ValueError):
        ScriptStep()
    with pytest.raises(ValueError):
        ScriptStep(failure="unknown")


def test_labels_must_agree_with_themselves():
    with pytest.raises(ValueError):
        AgentExpected(stop=AgentStopReason.ANSWERED, output_nature=OutputNature.SYNTHETIC, model_turns=1)
    with pytest.raises(ValueError):
        AgentExpected(stop=AgentStopReason.NO_RELEVANT_CLAIM, output_nature=OutputNature.SYNTHETIC, model_turns=1)
    with pytest.raises(ValueError):
        AgentCase(case_id="Upper", group="g", fixture="f", question=None, expected=None)


def test_the_selection_suites_are_unchanged_inputs(dataset):
    assert dataset.source.version == "copilot-eval/v2"
    assert len(dataset.source.suite(Suite.SELECTION)) == 88
    assert len(dataset.source.suite(Suite.CONTRACT)) == 18


def test_cli_writes_the_report(tmp_path, capsys):
    output = tmp_path / "report.json"
    assert copilot_agent_eval.main(["--dataset", str(DATASET_DIR), "--output", str(output)]) == 0
    assert output.read_text(encoding="utf-8") == capsys.readouterr().out


def test_the_cli_has_no_way_to_reach_a_model():
    parser_source = Path(copilot_agent_eval.__file__).read_text(encoding="utf-8")
    assert "allow-paid" not in parser_source
    assert "AnthropicAgentModel" not in parser_source
    assert "evaluation.live" not in parser_source
    with pytest.raises(SystemExit):
        copilot_agent_eval.main(["--dataset", str(DATASET_DIR), "--allow-paid-calls"])


def test_the_api_and_worker_never_load_the_agent():
    code = (
        "import sys; import footballai_v2.api; import footballai_v2.worker; import footballai_v2.copilot; "
        "loaded = [m for m in sys.modules if m.startswith('footballai_v2.copilot.agent')]; "
        "sys.exit(1 if loaded else 0)"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "v2" / "src")}
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True)
    assert result.returncode == 0, result.stderr.decode()


def test_the_evaluation_runs_without_the_sdk():
    code = (
        "import sys; from pathlib import Path; "
        "from footballai_v2.cli.copilot_agent_eval import build_report; "
        "from footballai_v2.copilot.evaluation.agent_dataset import load_agent_dataset; "
        f"build_report(load_agent_dataset(Path({str(DATASET_DIR)!r}))); "
        "sys.exit(1 if 'anthropic' in sys.modules else 0)"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "v2" / "src")}
    env.pop("ANTHROPIC_API_KEY", None)
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True)
    assert result.returncode == 0, result.stderr.decode()


_ALLOWED_ROOTS = {"__future__", "dataclasses", "enum", "json", "pydantic", "re", "time", "typing", "uuid"}
_ALLOWED_PACKAGES = ("footballai_v2.contracts", "footballai_v2.copilot", "footballai_v2.storage.ports")


@pytest.mark.parametrize("path", sorted(AGENT_SOURCE.glob("*.py")), ids=lambda path: path.name)
def test_the_agent_imports_nothing_that_could_reach_outside_it(path):
    """No web framework, inference, Azure, subprocess, socket, or filesystem module."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert name.split(".")[0] in _ALLOWED_ROOTS or name.startswith(_ALLOWED_PACKAGES), name


@pytest.mark.parametrize("path", sorted(AGENT_SOURCE.glob("*.py")), ids=lambda path: path.name)
def test_the_agent_never_opens_files_or_executes_text(path):
    calls = {
        node.func.id
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not calls & {"open", "exec", "eval", "compile", "__import__"}
