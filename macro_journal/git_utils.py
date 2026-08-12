"""git_utils.py

Commits and pushes EOD Markdown reports to the configured GitHub repo using
a Personal Access Token. Uses GitPython against the local repo clone.
"""
import os
from pathlib import Path
from git import Repo, GitCommandError


def _repo_path() -> Path:
    path = os.environ.get("LOCAL_REPO_PATH")
    if not path:
        raise RuntimeError("LOCAL_REPO_PATH not set in .env")
    return Path(path)


def commit_and_push_eod(file_path: Path, day: str) -> str:
    """
    Copies/uses file_path (already inside the repo, e.g. data/eod/DAY.md),
    commits it, and pushes using the configured PAT.
    Returns a status message.
    """
    repo_path = _repo_path()
    repo = Repo(repo_path)

    user_name = os.environ.get("GIT_USER_NAME", "cclimon")
    user_email = os.environ.get("GIT_USER_EMAIL", "cc@centilepartners.com")
    with repo.config_writer() as cw:
        cw.set_value("user", "name", user_name)
        cw.set_value("user", "email", user_email)

    rel_path = file_path.relative_to(repo_path) if file_path.is_absolute() else file_path
    repo.index.add([str(rel_path)])
    repo.index.commit(f"EOD journal entry: {day}")

    pat = os.environ.get("GITHUB_PAT")
    repo_slug = os.environ.get("GITHUB_REPO")
    if not pat or not repo_slug:
        return "Committed locally, but GITHUB_PAT / GITHUB_REPO not set — skipped push."

    origin_url = f"https://{pat}@github.com/{repo_slug}.git"
    try:
        if "origin" in [r.name for r in repo.remotes]:
            repo.remotes.origin.set_url(origin_url)
        else:
            repo.create_remote("origin", origin_url)
        repo.remotes.origin.push()
        return f"Committed and pushed EOD report for {day}."
    except GitCommandError as e:
        return f"Committed locally, but push failed: {e}"
