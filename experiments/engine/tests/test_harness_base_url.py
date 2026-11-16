"""Compatible-endpoint support of the editor (cryowisp.harness.launch: read_base_url, read_auth_token,
billing_record, ClaudeLauncher): point the Claude Code CLI at GLM, Kimi, DeepSeek or another Anthropic-compatible
endpoint, so the editor's model is the caller's choice and no Anthropic credential is needed."""

from __future__ import annotations

from pathlib import Path

import pytest

from cryowisp.harness import launch

SESSION = {"mcp": {"tool_timeout_ms": 1000, "connect_timeout_ms": 1000}}
TOKEN = "bearer-token-not-real-0123456789"
URL = "https://open.bigmodel.cn/api/anthropic"
KEY = "sk-test-not-a-real-key-0123456789"


def _file(tmp_path: Path, name: str, content: str, mode: int = 0o600) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    path.chmod(mode)
    return path


def _fake_claude(tmp_path: Path) -> str:
    exe = tmp_path / "claude"
    exe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    exe.chmod(0o755)
    return str(exe)


def test_the_base_url_is_read_from_a_private_one_line_file(tmp_path: Path) -> None:
    assert launch.read_base_url(_file(tmp_path, "base_url", URL + "\n")) == URL
    with pytest.raises(launch.BaseUrlFileError, match="https://"):
        launch.read_base_url(_file(tmp_path, "base_url", "open.bigmodel.cn/api/anthropic\n"))
    with pytest.raises(launch.BaseUrlFileError, match="does not exist"):
        launch.read_base_url(tmp_path / "absent")
    with pytest.raises(launch.BaseUrlFileError, match="regular file"):
        launch.read_base_url(tmp_path)


def test_the_auth_token_file_follows_the_same_privacy_rules(tmp_path: Path) -> None:
    path = _file(tmp_path, "auth_token", TOKEN + "\n")
    assert launch.read_auth_token(path) == TOKEN
    assert isinstance(launch.read_auth_token(path), str)
    with pytest.raises(launch.AuthTokenFileError, match="mode"):
        launch.read_auth_token(_file(tmp_path, "auth_token", TOKEN + "\n", mode=0o644))
    with pytest.raises(launch.AuthTokenFileError) as raised:
        launch.read_auth_token(_file(tmp_path, "auth_token", f"{TOKEN}\n{TOKEN}\n"))
    assert TOKEN not in str(raised.value)


def test_billing_record_holds_paths_never_secrets(tmp_path: Path) -> None:
    token = _file(tmp_path, "auth_token", TOKEN + "\n")
    url = _file(tmp_path, "base_url", URL + "\n")
    record = launch.billing_record(None, base_url_file=url, auth_token_file=token)
    assert record == {"mode": launch.BILLING_AUTH_TOKEN_FILE, "auth_token_file": str(token.absolute()),
                      "base_url_file": str(url.absolute())}
    assert TOKEN not in repr(record) and URL not in repr(record)
    key = _file(tmp_path, "api_key", KEY + "\n")
    assert launch.billing_record(key, base_url_file=url) == {
        "mode": launch.BILLING_API_KEY_FILE, "api_key_file": str(key.absolute()),
        "base_url_file": str(url.absolute())}
    assert launch.billing_record(None) == {"mode": launch.BILLING_SUBSCRIPTION}


def test_several_credential_files_are_refused(tmp_path: Path) -> None:
    token = _file(tmp_path, "auth_token", TOKEN + "\n")
    with pytest.raises(ValueError, match="not both"):
        launch.billing_record(_file(tmp_path, "api_key", KEY + "\n"), auth_token_file=token)
    with pytest.raises(ValueError, match="not both"):
        launch.ClaudeLauncher(_fake_claude(tmp_path), api_key_file=_file(tmp_path, "api_key", KEY + "\n"),
                              auth_token_file=token)


def test_the_launcher_sets_the_endpoint_and_token_in_the_child_only(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(launch.BASE_URL_ENV, "https://inherited.example.com")
    monkeypatch.setenv(launch.AUTH_TOKEN_ENV, "inherited-from-parent")
    plain = launch.ClaudeLauncher(_fake_claude(tmp_path))
    env, removed = plain.environment(SESSION)
    assert launch.BASE_URL_ENV not in env and launch.BASE_URL_ENV in removed
    assert launch.AUTH_TOKEN_ENV not in env and launch.AUTH_TOKEN_ENV in removed
    token = _file(tmp_path, "auth_token", TOKEN + "\n")
    url = _file(tmp_path, "base_url", URL + "\n")
    launcher = launch.ClaudeLauncher(_fake_claude(tmp_path), auth_token_file=token, base_url_file=url)
    env, removed = launcher.environment(SESSION)
    assert env[launch.BASE_URL_ENV] == URL and env[launch.AUTH_TOKEN_ENV] == TOKEN
    assert launcher.billing["mode"] == launch.BILLING_AUTH_TOKEN_FILE
    assert TOKEN not in repr(launcher.billing) and URL not in repr(launcher.billing)


def test_a_base_url_file_combines_with_an_api_key_file(tmp_path: Path) -> None:
    url = _file(tmp_path, "base_url", URL + "\n")
    key = _file(tmp_path, "api_key", KEY + "\n")
    launcher = launch.ClaudeLauncher(_fake_claude(tmp_path), api_key_file=key, base_url_file=url)
    env, _ = launcher.environment(SESSION)
    assert env[launch.BASE_URL_ENV] == URL and env[launch.API_KEY_ENV] == KEY
    assert launcher.billing["base_url_file"] == str(url.absolute())
    assert launcher.billing["mode"] == launch.BILLING_API_KEY_FILE


def test_a_bad_file_is_refused_before_any_session_is_prepared(tmp_path: Path) -> None:
    with pytest.raises(launch.BaseUrlFileError):
        launch.ClaudeLauncher(_fake_claude(tmp_path), base_url_file=_file(tmp_path, "base_url", URL + "\n",
                                                                          mode=0o644))
    with pytest.raises(launch.BaseUrlFileError, match="https://"):
        launch.ClaudeLauncher(_fake_claude(tmp_path), base_url_file=_file(tmp_path, "base_url", "not-a-url\n"))
    with pytest.raises(launch.AuthTokenFileError, match="mode"):
        launch.ClaudeLauncher(_fake_claude(tmp_path), auth_token_file=_file(tmp_path, "auth_token", TOKEN + "\n",
                                                                            mode=0o644))


def test_the_cli_takes_the_new_flags() -> None:
    import subprocess
    import sys
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_editor_session.py"
    proc = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert "--auth-token-file" in proc.stdout and "--base-url-file" in proc.stdout
    proc = subprocess.run([sys.executable, str(script), "--api-key-file", "/nonexistent/a",
                           "--auth-token-file", "/nonexistent/b", "--help"], capture_output=True, text=True)
    assert proc.returncode == 2 and "not allowed with argument" in proc.stderr
