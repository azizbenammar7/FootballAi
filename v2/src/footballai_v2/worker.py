"""Environment-configured local queue worker entrypoint."""

from __future__ import annotations

import logging
import os
import signal
import socket
import time

from footballai_v2.execution.coordinator import ExecutionSettings
from footballai_v2.execution.executor import AnalysisExecutor
from footballai_v2.execution.queue import LocalFilesystemQueue
from footballai_v2.storage import LocalAnalysisRunStore


def main() -> None:
    logging.basicConfig(level=os.getenv("FOOTBALLAI_LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = ExecutionSettings.from_environment()
    store = LocalAnalysisRunStore(settings.run_root); queue = LocalFilesystemQueue(settings.queue_root)
    worker_id = os.getenv("FOOTBALLAI_WORKER_ID", f"{socket.gethostname()}-{os.getpid()}")[:128]
    poll = max(.05, float(os.getenv("FOOTBALLAI_WORKER_POLL_SECONDS", ".25")))
    claim_timeout = float(os.getenv("FOOTBALLAI_JOB_CLAIM_TIMEOUT_SECONDS", "300"))
    delay = float(os.getenv("FOOTBALLAI_DEMO_STAGE_DELAY_SECONDS", ".12"))
    stopped = False
    def stop(_signum, _frame):
        nonlocal stopped; stopped = True
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    queue.recover_abandoned(claim_timeout, store)
    executor = AnalysisExecutor(store, stage_delay_seconds=delay)
    logging.getLogger("footballai_v2.worker").info("worker_started worker_id=%s", worker_id)
    while not stopped:
        job = queue.claim(worker_id)
        if job is None:
            time.sleep(poll); continue
        status = executor.execute(job, worker_id)
        if status.value in {"succeeded", "partial"}: queue.complete(job)
        elif status.value == "cancelled": queue.cancel(job.run_id)
        else: queue.fail(job)


if __name__ == "__main__":
    main()
