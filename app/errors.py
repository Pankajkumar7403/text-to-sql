"""User-facing inference errors (safe for UI and API responses)."""


class InferenceUserError(Exception):
    """Raised when SQL generation fails; message is safe to show clients."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
