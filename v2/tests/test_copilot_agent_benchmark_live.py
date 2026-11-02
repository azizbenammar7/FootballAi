"""M3-A: the live runner, end to end, against a local server and the pinned SDK.

This is the whole paid path: approval, preflight, budget, adapter, scorer, report. It
sends everything to 127.0.0.1, so it spends nothing and measures no model, and the
report it produces says so.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("anthropic")

from footballai_v2.cli import copilot_agent_benchmark as cli
from footballai_v2.copilot.evaluation import agent_live as live_module
from footballai_v2.copilot.evaluation.agent_benchmark import Split, load_benchmark
from footballai_v2.copilot.evaluation.agent_rule import load_rule

from test_copilot_agent_sdk_contract import reply
from test_copilot_sdk_contract import CANARY, MODEL, error_body, local_api  # noqa: F401  (a fixture)


DATASET_DIR = Path(__file__).resolve().parents[2] / "evaluation" / "copilot" / "agent-v2"
EMPTY_FINAL = reply([{"type": "text", "text": '{"claim_ids": []}'}])


@pytest.fixture
def paid(local_api, monkeypatch, tmp_path):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    local_api.routes["/v1/models"] = (200, MODEL)
    local_api.routes["/v1/messages/count_tokens"] = (200, {"input_tokens": 500})
    dataset = load_benchmark(DATASET_DIR)
    _rule, rule_sha = load_rule(DATASET_DIR)

    def record(**changes):
        values = {
            "schema": "footballai.copilot-agent-live-approval/v1",
            "status": "approved",
            "approved_by": "test",
            "approved_on": "2026-11-02",
            "stage": "readiness",
            "split": "dev",
            "model": "claude-haiku-5-5",
            "max_model_requests": 200,
            "max_cost_usd": 2.0,
            "decision_rule_sha256": rule_sha,
            "cases_sha256": dataset.sha256(Split.DEV),
            **changes,
        }
        path = tmp_path / "approval.json"
        path.write_text(json.dumps(values), encoding="utf-8")
        return str(path)

    return record


def run(capsys, approval, *extra, requests="200", cost="2.0", output_path=None):
    path = output_path or Path(approval).with_name("report.json")
    code = cli.main(
        ["--dataset", str(DATASET_DIR), "--allow-paid-calls", "--approval", approval, "--max-requests", requests,
         "--max-cost-usd", cost, "--output", str(path), *extra]
    )
    output = capsys.readouterr().out
    return code, json.loads(output) if output.startswith("{") else output


def messages_sent(local_api):
    return [request for request in local_api.requests if request[1] == "/v1/messages"]


def test_the_whole_paid_path_runs_and_labels_itself_a_test_double(local_api, paid, capsys):
    local_api.body = EMPTY_FINAL
    code, report = run(capsys, paid())
    assert code == 0
    assert report["mode"] == "live"
    assert report["endpoint"] == "loopback_test_double"
    assert report["real_model_quality_measured"] is False
    assert report["decision_rule"]["verdict"] == "NOT_A_MODEL_EVALUATION"
    assert report["decision_rule"]["criteria"] == []
    assert report["live_model_requests"] == len(messages_sent(local_api)) == report["questions_asked"]
    assert report["questions_asked"] == report["questions_planned"] == 31


def test_a_report_holds_no_prompt_question_answer_or_key(local_api, paid, capsys):
    local_api.body = EMPTY_FINAL
    _code, report = run(capsys, paid())
    text = json.dumps(report)
    assert CANARY not in text
    assert "What is the status" not in text
    assert "You retrieve and select evidence" not in text


def test_a_report_separates_actual_usage_from_its_cost_estimate(local_api, paid, capsys):
    local_api.body = EMPTY_FINAL
    _code, report = run(capsys, paid())
    tokens, cost = report["operations"]["tokens"], report["operations"]["cost"]
    assert tokens["usage_sources"]["provider_reported"] == 31
    assert tokens["actual_input_tokens"] == 31 * 800
    assert "not a bill" in cost["basis"]
    assert report["budget"]["guard_charged_usd"] == pytest.approx(cost["estimated_total_usd"], abs=1e-6)


def test_each_case_row_carries_its_correlation_and_provider_ids(local_api, paid, capsys):
    local_api.body = EMPTY_FINAL
    _code, report = run(capsys, paid())
    row = report["cases"][0]
    assert row["correlation_id"] and row["provider_request_ids"] == ["req_local"]
    assert len({case["correlation_id"] for case in report["cases"]}) == 31


def test_first_use_schemas_are_identified(local_api, paid, capsys):
    local_api.body = EMPTY_FINAL
    _code, report = run(capsys, paid())
    latency = report["operations"]["latency"]
    assert latency["end_to_end_first_use_schema"]["n"] == 19
    assert latency["end_to_end_repeat_schema"]["n"] == 12


def test_the_request_cap_stops_the_run_with_a_partial_report(local_api, paid, capsys):
    local_api.body = EMPTY_FINAL
    code, report = run(capsys, paid(), requests="3")
    assert code == 4
    assert report["stop_reason"] == "budget_exhausted"
    assert len(messages_sent(local_api)) == 3


def test_the_money_cap_stops_the_run_before_sending(local_api, paid, capsys):
    local_api.body = EMPTY_FINAL
    code, report = run(capsys, paid(), cost="0.0001")
    assert code == 4
    assert report["stop_reason"] == "budget_exhausted"
    assert messages_sent(local_api) == []


def test_an_authentication_failure_stops_at_once(local_api, paid, capsys):
    local_api.status = 401
    local_api.body = error_body("authentication_error")
    code, report = run(capsys, paid())
    assert code == 4
    assert report["stop_reason"] == "fatal_failure:authentication"
    assert len(messages_sent(local_api)) == 1


def test_three_consecutive_failures_stop_the_run(local_api, paid, capsys):
    local_api.status = 500
    local_api.body = error_body("api_error")
    code, report = run(capsys, paid())
    assert code == 4
    assert report["stop_reason"] == "consecutive_failures"
    assert len(messages_sent(local_api)) == 3


def test_a_failed_preflight_sends_no_generation(local_api, paid, capsys):
    local_api.routes["/v1/models"] = (404, error_body("not_found_error"))
    code, output = run(capsys, paid())
    assert code == 3
    assert "preflight failed" in output
    assert messages_sent(local_api) == []


def test_a_retired_model_fails_preflight(local_api, paid, capsys):
    local_api.routes["/v1/models"] = (200, {**MODEL, "lifecycle": "retired"})
    code, _output = run(capsys, paid())
    assert code == 3
    assert messages_sent(local_api) == []


def test_a_preflight_alone_sends_no_generation(local_api, paid, capsys):
    code = cli.main(["--dataset", str(DATASET_DIR), "--preflight"])
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert messages_sent(local_api) == []
    assert report["preflight"]["counted_requests"] == 31
    assert report["plan"]["questions_reaching_a_model"] == 31


def test_a_changed_case_file_invalidates_the_approval(local_api, paid, capsys):
    code, output = run(capsys, paid(cases_sha256="0" * 64))
    assert code == 2
    assert "approval_other_cases" in output
    assert local_api.requests == []


def test_nothing_is_sent_before_every_check_passes(local_api, paid, capsys):
    code, _output = run(capsys, paid(status="not_approved"))
    assert code == 2
    assert local_api.requests == []


def test_questions_the_router_stops_still_count_in_every_denominator(local_api, paid, capsys):
    local_api.body = EMPTY_FINAL
    _code, report = run(capsys, paid())
    assert report["questions_stopped_before_a_model"] == 19
    assert report["metrics"]["cases"] == 50  # 31 asked, and 19 no model could be asked
    assert report["live_model_requests"] == 31  # the 19 cost nothing


def test_a_paid_run_must_keep_its_report(local_api, paid, capsys):
    code = cli.main(["--dataset", str(DATASET_DIR), "--allow-paid-calls", "--approval", paid(), "--max-requests", "5", "--max-cost-usd", "0.1"])
    assert code == 2
    assert "output_required" in capsys.readouterr().out
    assert local_api.requests == []


def test_a_paid_run_never_overwrites_a_report(local_api, paid, capsys, tmp_path):
    existing = tmp_path / "already-here.json"
    existing.write_text("{}", encoding="utf-8")
    code, output = run(capsys, paid(), output_path=existing)
    assert code == 2
    assert "output_exists" in output
    assert existing.read_text(encoding="utf-8") == "{}"
    assert local_api.requests == []


def test_a_decision_report_has_one_fixed_home_so_it_cannot_be_rerun_elsewhere(local_api, paid, capsys, monkeypatch):
    monkeypatch.setattr(cli, "check_approval", lambda *_args, **_kwargs: None)
    code, output = run(capsys, paid(), "--split", "heldout")
    assert code == 2
    assert "decision_report_path" in output
    assert local_api.requests == []


def test_a_defect_mid_run_still_writes_the_report(local_api, paid, capsys, monkeypatch):
    local_api.body = EMPTY_FINAL
    real = live_module.run_case
    calls = []

    def fails_on_the_third(*args, **kwargs):
        calls.extend([1] if kwargs.get("real_time") else [])  # planning uses run_case too
        if len(calls) == 3:
            raise RuntimeError("a scorer defect")
        return real(*args, **kwargs)

    monkeypatch.setattr(live_module, "run_case", fails_on_the_third)
    code, report = run(capsys, paid())
    assert code == 4
    assert report["stop_reason"] == "runner_error"
    assert report["questions_asked"] == 2


def test_an_interrupt_still_writes_the_report(local_api, paid, capsys, monkeypatch):
    local_api.body = EMPTY_FINAL
    real = live_module.run_case
    calls = []

    def interrupted(*args, **kwargs):
        calls.extend([1] if kwargs.get("real_time") else [])
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(*args, **kwargs)

    monkeypatch.setattr(live_module, "run_case", interrupted)
    code, report = run(capsys, paid())
    assert code == 4
    assert report["stop_reason"] == "interrupted"
