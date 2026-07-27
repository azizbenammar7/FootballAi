.PHONY: v2-demo v2-test v2-v1-compat-setup v2-v1-compat-readiness v2-v1-compat-smoke v2-demo-v1-compat

v2-demo:
	./v2/dev/run_demo.sh

v2-test:
	./v2/dev/run_tests.sh

v2-v1-compat-setup:
	./v2/dev/setup_v1_compat.sh

v2-v1-compat-readiness:
	PYTHONPATH=v2/src $${FOOTBALLAI_V2_PYTHON:-.venv-test/bin/python} -m footballai_v2.cli.v1_compat_runtime check

v2-v1-compat-smoke:
	./v2/dev/run_v1_compat_smoke.sh

v2-demo-v1-compat:
	./v2/dev/run_v1_compat_demo.sh
