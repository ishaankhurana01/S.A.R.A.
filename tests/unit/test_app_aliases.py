"""Unit tests for automation.app_aliases.resolve_alias."""

from __future__ import annotations

import pytest

from automation.app_aliases import resolve_alias


@pytest.mark.parametrize(
    "name,os_key,expected",
    [
        ("VS Code", "Windows", "code"),
        ("VS Code", "Darwin", "Visual Studio Code"),
        ("VS Code", "Linux", "code"),
        ("vs code", "Darwin", "Visual Studio Code"),  # case-insensitive
        ("  VS   Code  ", "Darwin", "Visual Studio Code"),  # whitespace-normalized
        ("Chrome", "Windows", "chrome"),
        ("Chrome", "Darwin", "Google Chrome"),
        ("Chrome", "Linux", "google-chrome"),
        ("Terminal", "Darwin", "Terminal"),
        ("Calculator", "Darwin", "Calculator"),
        ("Calculator", "Windows", "calc"),
        ("Calculator", "Linux", "gnome-calculator"),
    ],
)
def test_resolve_alias_known_names(name: str, os_key: str, expected: str) -> None:
    assert resolve_alias(name, os_key=os_key) == expected


def test_resolve_alias_unknown_name_returns_input_unchanged() -> None:
    assert resolve_alias("SomeObscureApp", os_key="Darwin") == "SomeObscureApp"


def test_resolve_alias_unknown_name_strips_whitespace() -> None:
    assert resolve_alias("  SomeObscureApp  ", os_key="Darwin") == "SomeObscureApp"


def test_resolve_alias_unsupported_os_key_falls_back_to_stripped_name() -> None:
    # A known alias but an os_key with no entry for it shouldn't happen in
    # practice (only Windows/Darwin/Linux are ever passed), but the
    # function should degrade gracefully rather than raise.
    assert resolve_alias("VS Code", os_key="FreeBSD") == "VS Code"
