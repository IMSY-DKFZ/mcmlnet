"""Helpers for runtime environment configuration."""

import os


def require_env(name: str) -> str:
    """Return a required environment variable or raise a clear runtime error."""
    value = os.getenv(name)
    if value:
        return value

    raise RuntimeError(
        f"Environment variable '{name}' is not set. "
        "Load your .env at the application boundary or set it explicitly "
        "before calling mcmlnet helpers that require it."
    )
