"""Sovereign financing methodology page (Equations-style)."""
from .page import build
from .registry import EQUATIONS, SECTIONS

write = build
__all__ = ["EQUATIONS", "SECTIONS", "build", "write"]
