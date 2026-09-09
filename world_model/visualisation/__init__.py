"""Local visual diagnostics and prior/posterior demo rendering.

Heavy plotting dependencies stay lazy so the static HTML progress and artifact
management commands do not initialise Matplotlib or its font cache.
"""

from __future__ import annotations

from typing import Any

__all__ = ["create_demo"]


def __getattr__(name: str) -> Any:
    if name == "create_demo":
        from world_model.visualisation.animation import create_demo

        return create_demo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
