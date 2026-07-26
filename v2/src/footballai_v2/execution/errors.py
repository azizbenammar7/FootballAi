"""Safe public execution errors."""


class ExecutionFailure(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message
        self.retryable = retryable


class CancellationObserved(RuntimeError):
    pass
