from __future__ import annotations

__all__ = ["create_app"]

__version__ = "0.1.0"


def create_app(*args, **kwargs):
    """Create the Dash app, importing UI dependencies only when requested."""
    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)
