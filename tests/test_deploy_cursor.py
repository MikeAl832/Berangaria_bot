"""Exercise deployment safety without touching Docker or production state."""

import os
from pathlib import Path
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "deploy-cursor.sh"
SHA = "a" * 40
MOCK_COMMAND = r'''#!/usr/bin/env python3
import os
from pathlib import Path
import sys

name, args = Path(sys.argv[0]).name, sys.argv[1:]
with open(os.environ["MOCK_CALLS"], "a") as stream:
    stream.write(name + " " + " ".join(args) + "\n")
if name == "hostname":
    print(os.environ.get("MOCK_HOST", "cursor"))
elif name == "id":
    print("box")
elif name == "git" and args[:1] == ["rev-parse"]:
    print(os.environ["DEPLOY_SHA"])
elif name == "docker":
    if "build" in args and os.environ.get("MOCK_BUILD_FAIL"):
        sys.exit(1)
    if "ps" in args and "-q" in args:
        print("bot-container")
    elif args[:1] == ["logs"]:
        print("Бот запущен!")
        print("PRIVATE CHAT CONTENT")
    elif args[:1] == ["inspect"]:
        field = args[2]
        if "StartedAt" in field:
            print("2026-09-14T00:00:00Z")
        elif "RestartCount" in field:
            print("0")
        elif "Status" in field:
            print(os.environ.get("MOCK_STATUS", "running"))
    if "config" in args:
        # Missing forwarded secrets must not mask .env's existing values.
        assert "TELEGRAM_API_ID" not in os.environ
'''


@pytest.fixture
def deployment(tmp_path):
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    for name in ("hostname", "id", "git", "docker", "sleep"):
        command = binary_dir / name
        command.write_text(MOCK_COMMAND)
        command.chmod(0o755)
    for directory in (".git", "bot_data", "qdrant_storage"):
        (tmp_path / directory).mkdir()
    (tmp_path / "bot_data" / "bot_state.db").write_bytes(b"existing state")
    (tmp_path / ".env").write_text("TELEGRAM_API_ID=existing-id\nFISH_API_KEY=old\n")
    environment = {
        **os.environ,
        "PATH": f"{binary_dir}:{os.environ['PATH']}",
        "DEPLOY_SHA": SHA,
        "DEPLOY_TEST_DIR": str(tmp_path),
        "MOCK_CALLS": str(tmp_path / "calls"),
        "TELEGRAM_API_ID": "",
        "TELEGRAM_API_HASH": "",
        "FISH_API_KEY": "",
        "FISH_VOICE_ID": "",
        "USER_BRIDGE_SESSION": "",
        "OPENROUTER_API_KEY": "",
        "XAI_API_KEY": "",
    }

    def run(**overrides):
        # Production's absolute directory checks are mapped to this fixture.
        harness = '''
cd() { builtin cd "$DEPLOY_TEST_DIR"; }
test() {
  if [[ "$*" == "-d /var/lib/telegram-bot-api" ]]; then return 0; fi
  builtin test "$@"
}
export -f cd test
bash "$1"
'''
        result = subprocess.run(
            ["bash", "-c", harness, "bash", str(SCRIPT)],
            env={**environment, **overrides},
            text=True,
            capture_output=True,
            check=False,
        )
        return result, (tmp_path / "calls").read_text()

    return tmp_path, run


def test_deploy_preserves_existing_secrets_and_checks_exact_commit(deployment):
    directory, run = deployment
    result, calls = run()
    assert result.returncode == 0, result.stderr
    assert (directory / ".env").read_text() == "TELEGRAM_API_ID=existing-id\nFISH_API_KEY=old\n"
    assert f"git fetch origin {SHA}" in calls
    assert f"git checkout --detach {SHA}" in calls
    assert "git reset" not in calls
    assert "PRIVATE CHAT CONTENT" not in result.stdout + result.stderr
    assert "--no-build --pull never --remove-orphans" in calls


def test_deploy_updates_secret_without_printing_it(deployment):
    directory, run = deployment
    result, _ = run(FISH_API_KEY="replacement-secret")
    assert result.returncode == 0, result.stderr
    assert "FISH_API_KEY=replacement-secret\n" in (directory / ".env").read_text()
    assert "replacement-secret" not in result.stdout + result.stderr
    assert (directory / ".env").stat().st_mode & 0o777 == 0o600


def test_wrong_host_cannot_deploy(deployment):
    _, run = deployment
    result, calls = run(MOCK_HOST="old-vps")
    assert result.returncode != 0
    assert "runner host is not cursor" in result.stdout
    assert "git " not in calls
    assert "docker " not in calls


def test_missing_database_cannot_deploy(deployment):
    directory, run = deployment
    (directory / "bot_data" / "bot_state.db").unlink()
    result, calls = run()
    assert result.returncode != 0
    assert "production database is missing or empty" in result.stdout
    assert "git " not in calls
    assert "docker " not in calls


def test_failed_build_does_not_replace_running_containers(deployment):
    _, run = deployment
    result, calls = run(MOCK_BUILD_FAIL="1")
    assert result.returncode != 0
    assert " up " not in calls


def test_stopped_bot_fails_without_publishing_private_logs(deployment):
    _, run = deployment
    result, _ = run(MOCK_STATUS="exited")
    assert result.returncode != 0
    assert "PRIVATE CHAT CONTENT" not in result.stdout + result.stderr
