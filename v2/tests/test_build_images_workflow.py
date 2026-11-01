"""Build images must verify every image before any image is published.

These tests run the workflow's real step scripts against stub docker and trivy
binaries, so they need no Docker daemon, Azure access, or registry.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest

WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "build-images.yml"
SHA = "a" * 40
IMAGES = ("frontend", "api", "worker")
PHASE_A = "Phase A:"
PHASE_B = "Phase B:"

STUB_DOCKER = """#!/usr/bin/env bash
echo "docker $*" >> "$CALLS"
image=$(printf '%s\\n' "$@" | grep -m1 'footballai-')
if [[ "${FAIL_GATE:-}" == "$1" && "$image" == *"footballai-${FAIL_IMAGE:-}:"* ]]; then exit 1; fi
if [[ "$1" == buildx ]]; then exit 0; fi
if [[ "$1 $2" == "image inspect" ]]; then
  case "$*" in
    *RepoDigests*) echo "registry/image@sha256:abc" ;;
    *) echo "${STUB_ARCH:-linux/amd64}" ;;
  esac
  exit 0
fi
if [[ "$1" == run && "$*" == *" -u" ]]; then
  [[ "$image" == *"footballai-${ROOT_IMAGE:-}:"* ]] && echo 0 || echo 1000
  exit 0
fi
exit 0
"""

STUB_TRIVY = """#!/usr/bin/env bash
echo "trivy $*" >> "$CALLS"
image="${@: -1}"
if [[ "$*" == *"--exit-code 1"* && "$image" == *"footballai-${TRIVY_FAIL_IMAGE:-}:"* ]]; then exit 1; fi
if [[ "$*" == *cyclonedx* ]]; then
  [[ "$image" == *"footballai-${SBOM_FAIL_IMAGE:-}:"* ]] && exit 1
  while [[ "$1" != --output ]]; do shift; done
  mkdir -p "$(dirname "$2")"; echo "{}" > "$2"
fi
exit 0
"""


def bash_4() -> str | None:
    for candidate in (shutil.which("bash"), "/opt/homebrew/bin/bash", "/usr/local/bin/bash"):
        if candidate and Path(candidate).exists():
            version = subprocess.run(
                [candidate, "-c", "echo ${BASH_VERSINFO[0]}"],
                capture_output=True, text=True, check=True,
            ).stdout
            if int(version) >= 4:
                return candidate
    return None


def step_names() -> list[str]:
    names = []
    for line in WORKFLOW.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- name:"):
            names.append(stripped.removeprefix("- name:").strip().strip('"'))
    return names


def step_script(prefix: str) -> str:
    lines = WORKFLOW.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if f"- name: \"{prefix}" in line)
    body = start + 2
    assert lines[start + 1].strip() == "run: |"
    end = next(
        (i for i in range(body, len(lines)) if lines[i].startswith("      - ")), len(lines)
    )
    return textwrap.dedent("\n".join(lines[body:end]))


def run_workflow_steps(tmp_path: Path, **failure: str) -> tuple[int, list[str]]:
    """Run Phase A, and Phase B only if Phase A passes, as Actions would."""
    bash = bash_4()
    if bash is None:
        pytest.skip("the workflow uses associative arrays, which need bash 4")
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    for name, content in (("docker", STUB_DOCKER), ("trivy", STUB_TRIVY)):
        path = stubs / name
        path.write_text(content, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    calls = tmp_path / "calls.log"
    calls.touch()
    environment = {
        **os.environ,
        "PATH": f"{stubs}:{os.environ['PATH']}",
        "CALLS": str(calls),
        "SHA": SHA,
        "ACR_LOGIN_SERVER": "registry.example",
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
        **failure,
    }
    code = 1
    for prefix in (PHASE_A, PHASE_B):
        code = subprocess.run(
            [bash, "-c", step_script(prefix)], cwd=tmp_path, env=environment,
            capture_output=True, text=True,
        ).returncode
        if code != 0:
            break
    return code, calls.read_text(encoding="utf-8").splitlines()


def test_every_image_passes_then_pushes_only_sha_tags_in_order(tmp_path):
    code, calls = run_workflow_steps(tmp_path)

    pushes = [call for call in calls if call.startswith("docker push")]
    assert code == 0
    assert pushes == [
        f"docker push registry.example/footballai-{name}:{SHA}" for name in IMAGES
    ]
    assert calls.index(pushes[0]) > max(
        index for index, call in enumerate(calls) if call.startswith("trivy")
    )
    assert sorted(path.name for path in (tmp_path / "sbom").iterdir()) == sorted(
        f"footballai-{name}.{SHA}.cdx.json" for name in IMAGES
    )
    assert not any(":latest" in call for call in calls)


@pytest.mark.parametrize("image", IMAGES)
@pytest.mark.parametrize(
    ("variable", "gate"),
    [
        pytest.param("FAIL_IMAGE", "buildx", id="build"),
        pytest.param("FAIL_IMAGE", "run", id="runtime-check"),
        pytest.param("ROOT_IMAGE", "", id="root-user"),
        pytest.param("TRIVY_FAIL_IMAGE", "", id="trivy-critical"),
        pytest.param("SBOM_FAIL_IMAGE", "", id="sbom"),
    ],
)
def test_any_failed_gate_on_any_image_prevents_every_push(tmp_path, image, variable, gate):
    code, calls = run_workflow_steps(tmp_path, FAIL_GATE=gate, **{variable: image})

    assert code != 0
    assert not any(call.startswith("docker push") for call in calls)


def test_a_failed_image_does_not_hide_failures_in_later_images(tmp_path):
    code, calls = run_workflow_steps(tmp_path, TRIVY_FAIL_IMAGE="frontend")

    assert code != 0
    for name in IMAGES:
        assert any(
            call.startswith("trivy") and "--exit-code 1" in call and f"footballai-{name}:" in call
            for call in calls
        )


def test_azure_login_and_push_run_only_after_verification_and_artifact_upload():
    names = step_names()
    verify = next(i for i, name in enumerate(names) if name.startswith(PHASE_A))
    publish = next(i for i, name in enumerate(names) if name.startswith(PHASE_B))

    assert verify < names.index("Upload SBOMs") < publish
    assert verify < names.index("Azure login via OIDC (build identity)") < publish
    assert verify < names.index("ACR login via AAD token (no admin user, no password)") < publish
    assert "docker push" not in step_script(PHASE_A)


def test_workflow_preserves_oidc_and_platform_invariants():
    content = WORKFLOW.read_text(encoding="utf-8")

    assert "id-token: write" in content
    assert "environment: staging-build" in content
    assert "--platform linux/amd64" in content
    assert "--severity CRITICAL --ignore-unfixed --exit-code 1" in content
    assert "password" not in step_script(PHASE_B)
    assert "secrets." not in content
