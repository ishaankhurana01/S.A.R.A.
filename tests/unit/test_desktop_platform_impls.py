"""
Unit tests for the per-OS ``open_application`` implementations
(``windows_platform``, ``macos_platform``, ``linux_platform``) and
``automation.platform_factory.get_platform``'s automatic OS selection.

Every test here mocks ``subprocess``/``platform.system`` — these tests run
on Linux in CI regardless of which platform's code path they're
exercising, and none of them should ever actually launch a real
application.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from automation.desktop_platform import DesktopPlatform
from automation.platform_factory import get_platform
from automation.platforms.linux_platform import LinuxDesktopPlatform
from automation.platforms.macos_platform import MacDesktopPlatform
from automation.platforms.windows_platform import WindowsDesktopPlatform
from utils.exceptions import ApplicationLaunchError, InvalidDesktopTargetError, UnsupportedPlatformError


# --------------------------------------------------------------------------- #
# platform_factory: automatic OS selection
# --------------------------------------------------------------------------- #
@patch("automation.platform_factory._platform_module.system")
def test_factory_selects_windows(mock_system) -> None:
    mock_system.return_value = "Windows"
    assert isinstance(get_platform(), WindowsDesktopPlatform)


@patch("automation.platform_factory._platform_module.system")
def test_factory_selects_macos(mock_system) -> None:
    mock_system.return_value = "Darwin"
    assert isinstance(get_platform(), MacDesktopPlatform)


@patch("automation.platform_factory._platform_module.system")
def test_factory_selects_linux(mock_system) -> None:
    mock_system.return_value = "Linux"
    assert isinstance(get_platform(), LinuxDesktopPlatform)


@patch("automation.platform_factory._platform_module.system")
def test_factory_raises_for_unsupported_os(mock_system) -> None:
    mock_system.return_value = "PlanNine"
    with pytest.raises(UnsupportedPlatformError):
        get_platform()


@patch("automation.platform_factory._platform_module.system")
def test_factory_passes_through_screenshot_directory(mock_system, tmp_path) -> None:
    mock_system.return_value = "Linux"
    platform = get_platform(screenshot_directory=str(tmp_path / "shots"))
    assert isinstance(platform, DesktopPlatform)
    assert platform._screenshot_directory == tmp_path / "shots"


# --------------------------------------------------------------------------- #
# WindowsDesktopPlatform.open_application
# --------------------------------------------------------------------------- #
@patch("automation.platforms.windows_platform.subprocess.Popen")
def test_windows_open_application_uses_alias(mock_popen) -> None:
    platform = WindowsDesktopPlatform()
    result = platform.open_application("VS Code")

    mock_popen.assert_called_once_with(["code"], shell=False)
    assert "code" in result


@patch("automation.platforms.windows_platform.subprocess.Popen")
def test_windows_open_application_falls_back_to_raw_name(mock_popen) -> None:
    platform = WindowsDesktopPlatform()
    platform.open_application("SomeUnknownApp")

    mock_popen.assert_called_once_with(["SomeUnknownApp"], shell=False)


@patch("automation.platforms.windows_platform.subprocess.Popen")
def test_windows_open_application_never_uses_shell_true(mock_popen) -> None:
    platform = WindowsDesktopPlatform()
    platform.open_application("notepad")

    _, kwargs = mock_popen.call_args
    assert kwargs.get("shell") is False


@patch("automation.platforms.windows_platform.subprocess.Popen")
def test_windows_open_application_raises_when_all_candidates_fail(mock_popen) -> None:
    mock_popen.side_effect = FileNotFoundError("no such file")
    platform = WindowsDesktopPlatform()

    with pytest.raises(ApplicationLaunchError):
        platform.open_application("TotallyMadeUpApp")


def test_windows_open_application_validates_input() -> None:
    platform = WindowsDesktopPlatform()
    with pytest.raises(InvalidDesktopTargetError):
        platform.open_application("app; rm -rf /")


# --------------------------------------------------------------------------- #
# MacDesktopPlatform.open_application
# --------------------------------------------------------------------------- #
@patch("automation.platforms.macos_platform.subprocess.run")
def test_macos_open_application_uses_open_dash_a(mock_run) -> None:
    mock_run.return_value = MagicMock(returncode=0)
    platform = MacDesktopPlatform()

    result = platform.open_application("Safari")

    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    assert args[0] == ["open", "-a", "Safari"]
    assert kwargs.get("shell") is False
    assert "Safari" in result


@patch("automation.platforms.macos_platform.subprocess.run")
def test_macos_open_application_raises_on_nonzero_exit(mock_run) -> None:
    mock_run.side_effect = subprocess.CalledProcessError(1, ["open", "-a", "Ghost"], stderr=b"not found")
    platform = MacDesktopPlatform()

    with pytest.raises(ApplicationLaunchError):
        platform.open_application("Ghost")


@patch("automation.platforms.macos_platform.subprocess.run")
def test_macos_open_application_raises_on_timeout(mock_run) -> None:
    mock_run.side_effect = subprocess.TimeoutExpired(cmd=["open"], timeout=15)
    platform = MacDesktopPlatform()

    with pytest.raises(ApplicationLaunchError):
        platform.open_application("SlowApp")


# --------------------------------------------------------------------------- #
# LinuxDesktopPlatform.open_application
# --------------------------------------------------------------------------- #
@patch("automation.platforms.linux_platform.subprocess.Popen")
def test_linux_open_application_uses_alias(mock_popen) -> None:
    platform = LinuxDesktopPlatform()
    result = platform.open_application("VS Code")

    mock_popen.assert_called_once_with(["code"], shell=False)
    assert "code" in result


@patch("automation.platforms.linux_platform.subprocess.Popen")
def test_linux_open_application_guesses_hyphenated_command(mock_popen) -> None:
    platform = LinuxDesktopPlatform()
    platform.open_application("Some New App")

    mock_popen.assert_called_once_with(["some-new-app"], shell=False)


@patch("automation.platforms.linux_platform.subprocess.Popen")
def test_linux_open_application_raises_when_not_found(mock_popen) -> None:
    mock_popen.side_effect = FileNotFoundError("not found")
    platform = LinuxDesktopPlatform()

    with pytest.raises(ApplicationLaunchError):
        platform.open_application("NoSuchApp")


# --------------------------------------------------------------------------- #
# Phase 4 polish: centralized alias resolution across all three platforms
# --------------------------------------------------------------------------- #
@patch("automation.platforms.windows_platform.subprocess.Popen")
def test_windows_open_application_resolves_chrome_alias(mock_popen) -> None:
    WindowsDesktopPlatform().open_application("Chrome")
    mock_popen.assert_called_once_with(["chrome"], shell=False)


@patch("automation.platforms.macos_platform.subprocess.run")
def test_macos_open_application_resolves_vs_code_alias(mock_run) -> None:
    mock_run.return_value = MagicMock(returncode=0)
    result = MacDesktopPlatform().open_application("VS Code")

    args, _ = mock_run.call_args
    assert args[0] == ["open", "-a", "Visual Studio Code"]
    assert "Visual Studio Code" in result


@patch("automation.platforms.macos_platform.subprocess.run")
def test_macos_open_application_resolves_chrome_alias(mock_run) -> None:
    mock_run.return_value = MagicMock(returncode=0)
    MacDesktopPlatform().open_application("Chrome")

    args, _ = mock_run.call_args
    assert args[0] == ["open", "-a", "Google Chrome"]


@patch("automation.platforms.macos_platform.subprocess.run")
def test_macos_open_application_unknown_name_passed_through(mock_run) -> None:
    mock_run.return_value = MagicMock(returncode=0)
    MacDesktopPlatform().open_application("SomeThirdPartyApp")

    args, _ = mock_run.call_args
    assert args[0] == ["open", "-a", "SomeThirdPartyApp"]


@patch("automation.platforms.linux_platform.subprocess.Popen")
def test_linux_open_application_resolves_chrome_alias(mock_popen) -> None:
    LinuxDesktopPlatform().open_application("Chrome")
    mock_popen.assert_called_once_with(["google-chrome"], shell=False)


@patch("automation.platforms.windows_platform.subprocess.Popen")
def test_windows_open_application_falls_back_when_alias_launch_fails(mock_popen) -> None:
    # First candidate (alias "code") fails, second (raw "VS Code") also
    # fails -> both attempted, in order, before raising.
    mock_popen.side_effect = FileNotFoundError("not found")
    with pytest.raises(ApplicationLaunchError):
        WindowsDesktopPlatform().open_application("VS Code")

    calls = [c.args[0] for c in mock_popen.call_args_list]
    assert calls == [["code"], ["VS Code"]]
