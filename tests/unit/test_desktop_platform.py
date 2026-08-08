"""
Unit tests for ``automation.desktop_platform``: the abstract base's shared
methods (close_application, open_url, list_processes, current_directory,
take_screenshot) and the ``_validate_target``/``_validate_url`` safety
helpers.

``list_processes`` and ``current_directory`` are exercised against the
real OS (they're read-only and safe anywhere). ``close_application``,
``open_url``, and ``take_screenshot`` mock their external dependency
(``psutil``, ``webbrowser``, ``mss``) so behavior is deterministic
regardless of what's actually running/installed in the test environment.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from automation.desktop_platform import DesktopPlatform, _validate_target, _validate_url
from utils.exceptions import (
    ApplicationCloseError,
    ApplicationLaunchError,
    InvalidDesktopTargetError,
    ScreenshotCaptureError,
)


class _ConcretePlatform(DesktopPlatform):
    """Minimal concrete subclass — only open_application is abstract."""

    def open_application(self, name: str) -> str:
        return f"opened {name}"


@pytest.fixture
def platform(tmp_path) -> _ConcretePlatform:
    return _ConcretePlatform(screenshot_directory=tmp_path / "screenshots")


# --------------------------------------------------------------------------- #
# _validate_target / _validate_url
# --------------------------------------------------------------------------- #
def test_validate_target_accepts_normal_input() -> None:
    assert _validate_target("VS Code", field_name="application name") == "VS Code"


def test_validate_target_strips_whitespace() -> None:
    assert _validate_target("  Chrome  ", field_name="application name") == "Chrome"


@pytest.mark.parametrize("value", ["", "   ", None])
def test_validate_target_rejects_empty(value) -> None:
    with pytest.raises(InvalidDesktopTargetError):
        _validate_target(value, field_name="application name")


def test_validate_target_rejects_too_long() -> None:
    with pytest.raises(InvalidDesktopTargetError):
        _validate_target("x" * 400, field_name="application name")


@pytest.mark.parametrize(
    "dangerous",
    [
        "notepad; rm -rf /",
        "app | cat /etc/passwd",
        "app `whoami`",
        "app $(whoami)",
        "app && shutdown",
        "app || shutdown",
        "app\nnewline",
        "app > file.txt",
    ],
)
def test_validate_target_rejects_shell_metacharacters(dangerous: str) -> None:
    with pytest.raises(InvalidDesktopTargetError):
        _validate_target(dangerous, field_name="application name")


def test_validate_url_accepts_https() -> None:
    assert _validate_url("https://example.com") == "https://example.com"


def test_validate_url_prepends_https_for_bare_domain() -> None:
    assert _validate_url("example.com") == "https://example.com"


def test_validate_url_rejects_file_scheme() -> None:
    with pytest.raises(InvalidDesktopTargetError):
        _validate_url("file:///etc/passwd")


def test_validate_url_rejects_non_url_text() -> None:
    with pytest.raises(InvalidDesktopTargetError):
        _validate_url("this is not a url")


# --------------------------------------------------------------------------- #
# current_directory (real OS call — safe, read-only)
# --------------------------------------------------------------------------- #
def test_current_directory_reports_actual_cwd(platform: _ConcretePlatform) -> None:
    result = platform.current_directory()
    assert os.getcwd() in result


# --------------------------------------------------------------------------- #
# list_processes (real OS call — safe, read-only)
# --------------------------------------------------------------------------- #
def test_list_processes_returns_nonempty_summary(platform: _ConcretePlatform) -> None:
    result = platform.list_processes()
    assert "running application" in result


def test_list_processes_respects_limit(platform: _ConcretePlatform) -> None:
    result = platform.list_processes(limit=1)
    # With a limit of 1 and (almost certainly) more than 1 process
    # running, the "+N more" suffix should appear.
    assert "more not shown" in result or "1 running application" in result


# --------------------------------------------------------------------------- #
# close_application (mocked psutil)
# --------------------------------------------------------------------------- #
def _mock_process(name: str, pid: int = 1234, terminate_raises: Exception | None = None) -> MagicMock:
    proc = MagicMock()
    proc.info = {"name": name}
    proc.pid = pid
    if terminate_raises:
        proc.terminate.side_effect = terminate_raises
    return proc


@patch("automation.desktop_platform.psutil.process_iter")
def test_close_application_terminates_matching_processes(mock_iter, platform: _ConcretePlatform) -> None:
    mock_iter.return_value = [_mock_process("Code.exe"), _mock_process("chrome.exe")]

    result = platform.close_application("code")

    assert "Terminated 1 process" in result


@patch("automation.desktop_platform.psutil.process_iter")
def test_close_application_no_match_raises(mock_iter, platform: _ConcretePlatform) -> None:
    mock_iter.return_value = [_mock_process("chrome.exe")]

    with pytest.raises(ApplicationCloseError):
        platform.close_application("nonexistent-app-xyz")


@patch("automation.desktop_platform.psutil.process_iter")
def test_close_application_validates_input_first(mock_iter, platform: _ConcretePlatform) -> None:
    with pytest.raises(InvalidDesktopTargetError):
        platform.close_application("")
    mock_iter.assert_not_called()


@patch("automation.desktop_platform.psutil.process_iter")
def test_close_application_all_terminations_fail_raises(mock_iter, platform: _ConcretePlatform) -> None:
    import psutil as real_psutil

    mock_iter.return_value = [_mock_process("locked.exe", terminate_raises=real_psutil.AccessDenied())]

    with pytest.raises(ApplicationCloseError):
        platform.close_application("locked")


# --------------------------------------------------------------------------- #
# open_url (mocked webbrowser)
# --------------------------------------------------------------------------- #
@patch("automation.desktop_platform.webbrowser.open")
def test_open_url_success(mock_open, platform: _ConcretePlatform) -> None:
    mock_open.return_value = True

    result = platform.open_url("https://example.com")

    assert "Opened URL" in result
    mock_open.assert_called_once_with("https://example.com")


@patch("automation.desktop_platform.webbrowser.open")
def test_open_url_no_handler_raises(mock_open, platform: _ConcretePlatform) -> None:
    mock_open.return_value = False

    with pytest.raises(ApplicationLaunchError):
        platform.open_url("https://example.com")


@patch("automation.desktop_platform.webbrowser.open")
def test_open_url_exception_raises_launch_error(mock_open, platform: _ConcretePlatform) -> None:
    mock_open.side_effect = RuntimeError("no display")

    with pytest.raises(ApplicationLaunchError):
        platform.open_url("https://example.com")


def test_open_url_invalid_target_never_calls_webbrowser(platform: _ConcretePlatform) -> None:
    with patch("automation.desktop_platform.webbrowser.open") as mock_open:
        with pytest.raises(InvalidDesktopTargetError):
            platform.open_url("file:///etc/passwd")
        mock_open.assert_not_called()


# --------------------------------------------------------------------------- #
# take_screenshot (mocked mss)
# --------------------------------------------------------------------------- #
def test_take_screenshot_success(platform: _ConcretePlatform, tmp_path) -> None:
    fake_mss_module = MagicMock()
    fake_sct = MagicMock()
    fake_sct.monitors = [{"left": 0, "top": 0, "width": 100, "height": 100}]
    fake_sct.grab.return_value = MagicMock(rgb=b"\x00" * 30000, size=(100, 100))
    fake_mss_module.mss.return_value.__enter__.return_value = fake_sct

    save_path = str(tmp_path / "shot.png")
    with patch.dict("sys.modules", {"mss": fake_mss_module, "mss.tools": fake_mss_module.tools}):
        result = platform.take_screenshot(save_path)

    assert save_path in result
    fake_mss_module.tools.to_png.assert_called_once()


def test_take_screenshot_failure_raises_screenshot_capture_error(platform: _ConcretePlatform, tmp_path) -> None:
    fake_mss_module = MagicMock()
    fake_mss_module.mss.side_effect = RuntimeError("no display available")

    with patch.dict("sys.modules", {"mss": fake_mss_module, "mss.tools": MagicMock()}):
        with pytest.raises(ScreenshotCaptureError):
            platform.take_screenshot(str(tmp_path / "shot.png"))


def test_take_screenshot_default_path_uses_screenshot_directory(platform: _ConcretePlatform, tmp_path) -> None:
    fake_mss_module = MagicMock()
    fake_sct = MagicMock()
    fake_sct.monitors = [{"left": 0, "top": 0, "width": 100, "height": 100}]
    fake_sct.grab.return_value = MagicMock(rgb=b"\x00" * 30000, size=(100, 100))
    fake_mss_module.mss.return_value.__enter__.return_value = fake_sct

    with patch.dict("sys.modules", {"mss": fake_mss_module, "mss.tools": fake_mss_module.tools}):
        result = platform.take_screenshot(None)

    assert str(tmp_path / "screenshots") in result
