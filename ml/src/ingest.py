"""Add exported PocketStep sessions to the public dataset and push them.

Researchers export a session ZIP from the app (AirDrop puts it in ~/Downloads), then run:

    python -m ml.src.ingest                       # every S-*.zip in ~/Downloads
    python -m ml.src.ingest path/to/S-....zip     # specific ZIPs or session folders

Each session is validated before anything is published. New sessions are copied to
ml/data/raw/<session_id>/, committed and pushed. A re-export of an existing session with corrected
labels replaces its labels and metadata. A different recording that reuses a session ID is refused.
After the push, the GitHub Actions workflow retrains the published model (see ml/published/).

The data is published in a public repository, so the command asks for confirmation that every
participant consented to public release, unless --yes is given.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import schema
from .preprocessing import Session, load_session

DEFAULT_DATA_DIR = Path("ml/data/raw")
SESSION_FILES = (schema.SENSOR_FILE, schema.LABELS_FILE, schema.METADATA_FILE)
PARTICIPANT_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
MAX_UNCOMPRESSED_BYTES = 200 * 1024 * 1024

NEW, UPDATED, UNCHANGED, REJECTED = "new", "updated labels", "already published", "rejected"


@dataclass
class Candidate:
    source: Path
    folder: Optional[Path] = None
    session: Optional[Session] = None
    status: str = REJECTED
    problems: list[str] = field(default_factory=list)

    @property
    def session_id(self) -> str:
        return self.session.session_id if self.session else self.source.name


class IngestError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Reading exports
# ---------------------------------------------------------------------------

def extract_zip(zip_path: Path, destination: Path) -> Path:
    """Extracts only the three session files from an exported ZIP and returns the session folder.

    Member paths are checked so a crafted archive cannot write outside `destination`.
    """
    with zipfile.ZipFile(zip_path) as archive:
        members = [m for m in archive.infolist()
                   if not m.is_dir() and not m.filename.startswith("__MACOSX/")
                   and not Path(m.filename).name.startswith(".")]
        if sum(m.file_size for m in members) > MAX_UNCOMPRESSED_BYTES:
            raise IngestError(f"{zip_path.name} is larger than {MAX_UNCOMPRESSED_BYTES // 2**20} MB uncompressed")
        found: dict[str, zipfile.ZipInfo] = {}
        for member in members:
            parts = Path(member.filename).parts
            if member.filename.startswith("/") or ".." in parts or len(parts) > 2:
                raise IngestError(f"{zip_path.name} contains an unexpected path: {member.filename}")
            if parts[-1] in SESSION_FILES:
                if parts[-1] in found:
                    raise IngestError(f"{zip_path.name} contains more than one {parts[-1]}")
                found[parts[-1]] = member
        missing = [name for name in SESSION_FILES if name not in found]
        if missing:
            raise IngestError(f"{zip_path.name} is missing {', '.join(missing)}")
        folder = destination / zip_path.stem
        folder.mkdir(parents=True)
        for name, member in found.items():
            with archive.open(member) as src, open(folder / name, "wb") as dst:
                shutil.copyfileobj(src, dst)
    return folder


def files_equal(a: Path, b: Path) -> bool:
    return a.exists() and b.exists() and a.read_bytes() == b.read_bytes()


def assess(candidate: Candidate, data_dir: Path) -> Candidate:
    """Validates a session and decides whether it is new, an update, already published, or rejected."""
    session = load_session(candidate.folder)
    candidate.session = session
    if not session.report.ok:
        candidate.problems.append("failed validation:\n" + session.report.format())
        return candidate
    if session.synthetic:
        candidate.problems.append("is a synthetic session; only real recordings belong in the dataset")
        return candidate
    if not PARTICIPANT_PATTERN.match(session.participant_id):
        candidate.problems.append(f"participant_id {session.participant_id!r} is not an anonymous code "
                                  "(1-32 letters, digits, '-' or '_')")
        return candidate
    if not PARTICIPANT_PATTERN.match(session.session_id):
        candidate.problems.append(f"session_id {session.session_id!r} is not a valid folder name")
        return candidate

    existing = data_dir / session.session_id
    if not existing.exists():
        candidate.status = NEW
    elif not files_equal(existing / schema.SENSOR_FILE, candidate.folder / schema.SENSOR_FILE):
        candidate.problems.append(f"a different recording with session ID {session.session_id} is already "
                                  "published; refusing to overwrite it")
    elif all(files_equal(existing / n, candidate.folder / n) for n in SESSION_FILES):
        candidate.status = UNCHANGED
    else:
        candidate.status = UPDATED
    return candidate


def default_sources() -> list[Path]:
    downloads = Path.home() / "Downloads"
    return sorted(downloads.glob("S-*.zip")) if downloads.exists() else []


def collect(sources: list[Path], data_dir: Path, workdir: Path) -> list[Candidate]:
    candidates = []
    for source in sources:
        candidate = Candidate(source=source)
        try:
            if source.is_dir():
                if not (source / schema.SENSOR_FILE).exists():
                    raise IngestError(f"{source} has no {schema.SENSOR_FILE}")
                candidate.folder = source
            elif source.suffix.lower() == ".zip":
                candidate.folder = extract_zip(source, workdir)
            else:
                raise IngestError(f"{source} is neither a session ZIP nor a session folder")
        except (IngestError, zipfile.BadZipFile, OSError) as exc:
            candidate.problems.append(str(exc))
            candidates.append(candidate)
            continue
        candidates.append(assess(candidate, data_dir))

    # The same session given twice (e.g. a ZIP and its unzipped folder) is only published once.
    seen = set()
    for candidate in candidates:
        if candidate.status in (NEW, UPDATED):
            if candidate.session_id in seen:
                candidate.status = REJECTED
                candidate.problems.append("the same session ID appears more than once in this run")
            seen.add(candidate.session_id)
    return candidates


# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------

def git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    if result.returncode != 0:
        raise IngestError(f"git {' '.join(args)} failed:\n{result.stderr.strip() or result.stdout.strip()}")
    return result.stdout.strip()


def repo_root(start: Path) -> Path:
    return Path(git(start, "rev-parse", "--show-toplevel"))


def describe(candidate: Candidate) -> str:
    stats = candidate.session.report.stats
    m = candidate.session.metadata
    return (f"{candidate.session_id} (participant {candidate.session.participant_id}, {m.get('pocket_side')} pocket, "
            f"{stats.get('labels_left', 0)} left / {stats.get('labels_right', 0)} right / "
            f"{stats.get('labels_unknown', 0)} unknown, {stats.get('labels_invalid', 0)} invalid, "
            f"{stats.get('labels_unreviewed', 0)} unreviewed)")


def commit_message(publish: list[Candidate]) -> str:
    new = [c for c in publish if c.status == NEW]
    updated = [c for c in publish if c.status == UPDATED]
    participants = sorted({c.session.participant_id for c in publish})
    parts = []
    if new:
        parts.append(f"Add {len(new)} session(s)")
    if updated:
        parts.append(f"update labels of {len(updated)} session(s)")
    subject = " and ".join(parts) + f" ({', '.join(participants)})"
    body = ["Published with python -m ml.src.ingest after validation.", ""]
    body += [f"- {'new' if c.status == NEW else 'labels updated'}: {describe(c)}" for c in publish]
    return subject[0].upper() + subject[1:] + "\n\n" + "\n".join(body)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

CONSENT_PROMPT = (
    "These sessions will be published in a PUBLIC repository{remote}.\n"
    "Confirm that every participant consented to public release of their anonymized motion data,\n"
    "and that participant IDs and notes contain no names or other identifying details. Publish? [y/N] ")


def main(argv: Optional[list[str]] = None, input_fn=input) -> int:
    parser = argparse.ArgumentParser(description="Validate exported sessions, add them to the dataset and push.")
    parser.add_argument("sources", nargs="*", type=Path,
                        help="Session ZIPs or folders (default: every S-*.zip in ~/Downloads)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Dataset folder inside the repository")
    parser.add_argument("--yes", action="store_true", help="Skip the publication confirmation")
    parser.add_argument("--no-push", action="store_true", help="Commit locally but do not pull or push")
    parser.add_argument("--dry-run", action="store_true", help="Only validate and report; change nothing")
    args = parser.parse_args(argv)

    try:
        root = repo_root(Path.cwd())
    except IngestError:
        print("Run this command inside the pocketstep-ml repository.", file=sys.stderr)
        return 2
    data_dir = args.data_dir if args.data_dir.is_absolute() else root / args.data_dir
    sources = [s.expanduser().resolve() for s in args.sources] or default_sources()
    if not sources:
        print("No session ZIPs found. Export a session in the app (Export ZIP), AirDrop it to this Mac, "
              "then run this command again, or pass the ZIP path.")
        return 1

    try:
        if not (args.no_push or args.dry_run):
            print("Updating the repository (git pull)...")
            git(root, "pull", "--rebase", "--autostash")
        with tempfile.TemporaryDirectory() as workdir:
            candidates = collect(sources, data_dir, Path(workdir))
            return _publish(candidates, root, data_dir, args, input_fn)
    except IngestError as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        return 2


def _publish(candidates: list[Candidate], root: Path, data_dir: Path, args, input_fn) -> int:
    for c in candidates:
        if c.status == REJECTED:
            print(f"REJECTED  {c.source.name}: " + "; ".join(c.problems))
        elif c.status == UNCHANGED:
            print(f"skipped   {c.source.name}: {c.status}")
        else:
            print(f"{c.status.upper():<9} {describe(c)}")
            if c.session.report.stats.get("labels_unreviewed"):
                print("          note: unreviewed labels are not used for training until reviewed and re-exported")
            warnings = c.session.report.codes("warning")
            if warnings:
                print(f"          warnings: {', '.join(sorted(set(warnings)))}")
            notes = str(c.session.metadata.get("notes", "")).strip()
            if notes:
                print(f"          notes (will be public): {notes!r}")

    publish = [c for c in candidates if c.status in (NEW, UPDATED)]
    rejected = [c for c in candidates if c.status == REJECTED]
    if not publish:
        print("Nothing new to publish.")
        return 1 if rejected else 0
    if args.dry_run:
        print(f"Dry run: {len(publish)} session(s) would be published.")
        return 0

    if not args.yes:
        try:
            remote = git(root, "remote", "get-url", "origin")
        except IngestError:
            remote = ""
        answer = input_fn(CONSENT_PROMPT.format(remote=f" ({remote})" if remote else ""))
        if answer.strip().lower() not in ("y", "yes"):
            print("Nothing was published.")
            return 1

    paths = []
    for c in publish:
        destination = data_dir / c.session_id
        destination.mkdir(parents=True, exist_ok=True)
        for name in SESSION_FILES:
            shutil.copyfile(c.folder / name, destination / name)
            paths.append(str((destination / name).relative_to(root)))
    git(root, "add", "--", *paths)
    git(root, "commit", "-m", commit_message(publish), "--", *paths)
    print(f"Committed {len(publish)} session(s).")

    if args.no_push:
        print("Not pushed (--no-push). Run `git push` to publish.")
        return 0
    try:
        git(root, "push")
    except IngestError:
        # Someone else pushed meanwhile. Session folders never overlap, so a rebase is safe.
        git(root, "pull", "--rebase", "--autostash")
        git(root, "push")
    print("Pushed. GitHub Actions will now retrain the model; results appear in ml/published/README.md "
          "after a few minutes.")
    return 1 if rejected else 0


if __name__ == "__main__":
    sys.exit(main())
