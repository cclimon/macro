"""git_ops.py

Commits and (optionally) pushes EOD artifacts. Never raises — always returns
a human-readable status string, since a missing repo/remote/upstream or a
failed push must never fail the EOD itself.
"""
import os
import subprocess

ROOT = os.path.dirname(__file__)


def _run(args: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            args, cwd=ROOT, capture_output=True, text=True, timeout=30
        )
        ok = result.returncode == 0
        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        return ok, (err or out)
    except FileNotFoundError:
        return False, "git executable not found"
    except subprocess.TimeoutExpired:
        return False, "git command timed out"
    except Exception as e:  # noqa: BLE001 — deliberately broad, EOD must not fail
        return False, str(e)


def is_git_repo() -> bool:
    ok, _ = _run(["git", "rev-parse", "--is-inside-work-tree"])
    return ok


def commit_and_push(paths: list[str], message: str) -> str:
    """
    Returns one of:
      "skipped — not a git repository"
      "committed and pushed"
      "committed; push failed: <reason>"
      "commit failed: <reason>"
      "skipped push (EOD_PUSH=false); committed only"
    """
    if not is_git_repo():
        return "skipped — not a git repository"

    rel_paths = [os.path.relpath(p, ROOT) for p in paths]

    ok, msg = _run(["git", "add"] + rel_paths)
    if not ok:
        return f"commit failed: git add error: {msg}"

    ok, msg = _run(["git", "commit", "-m", message])
    if not ok:
        # "nothing to commit" is not fatal — treat as already committed
        if "nothing to commit" in msg.lower():
            pass
        else:
            return f"commit failed: {msg}"

    push_enabled = os.environ.get("EOD_PUSH", "true").lower() != "false"
    if not push_enabled:
        return "committed only (EOD_PUSH=false)"

    ok, msg = _run(["git", "push"])
    if ok:
        return "committed and pushed"
    return f"committed; push failed: {msg}"
