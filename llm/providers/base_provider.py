"""
LLM provider interface — locates the abstraction where the architecture
doc's folder layout (``llm/providers/base_provider.py``) expects it.

The actual ``LLMProvider`` ABC is defined in ``core.interfaces`` (it was
declared in Phase 1, before any concrete provider existed, per the
project's "declare the interface before the implementation" convention —
see that module's docstring). This module intentionally does not redefine
it: importing and re-exporting the single canonical definition avoids two
ABCs drifting out of sync, while still giving ``llm.providers.*`` modules
an in-package import path (``from llm.providers.base_provider import LLMProvider``)
that reads naturally alongside ``ollama_provider.py``.
"""

from __future__ import annotations

from core.interfaces import LLMProvider

__all__ = ["LLMProvider"]
