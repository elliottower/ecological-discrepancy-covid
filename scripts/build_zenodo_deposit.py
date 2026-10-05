"""Build the Zenodo archive for this repository.

Writes submission/zenodo/ecological-discrepancy-covid.zip from the tracked source,
excluding everything a public archive must not carry.  Refuses to overwrite an
existing archive, and asserts that no excluded path leaked in.

Run:  uv run --no-project python scripts/build_zenodo_deposit.py
"""
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "submission" / "zenodo" / "ecological-discrepancy-covid.zip"

# Excluded by directory prefix.  The manuscript is deposited as its own PDF, so
# paper/ adds nothing a reader needs and carries editorial correspondence and
# per-referee drafts.  SUBMISSIONS/ is journal correspondence too.  .results
# holds sealed run hashes that stay local.  zenodo_ny_hospitals and the CDC extract
# are third-party data available from their own repositories.
EXCLUDED_DIRS = (
    "paper/",
    "SUBMISSIONS/",
    "submission/",
    ".results/",
    "reviews/",
    "planning/",
    "data/zenodo_ny_hospitals/",
    "data/cdc_case_surveillance/",
    "dag_validation/data/",
)
EXCLUDED_NAMES = (".DS_Store",)
EXCLUDED_SUFFIXES = (".pyc", ".aux", ".log", ".out", ".bbl", ".blg", ".fdb_latexmk", ".fls")


def tracked_files():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    out.check_returncode()
    return [p for p in out.stdout.splitlines() if p]


def keep(path):
    if any(path.startswith(d) for d in EXCLUDED_DIRS):
        return False
    if Path(path).name in EXCLUDED_NAMES:
        return False
    if Path(path).suffix in EXCLUDED_SUFFIXES:
        return False
    return True


# paper/ is excluded wholesale, but paper/analyses/ holds the analysis pipeline,
# both frozen protocols and the computed results, which the deposit promises.
# Re-add it at a top-level path, the documented pattern for material under
# paper/ that a reader needs.
READD = {"paper/analyses/": "analyses/"}


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} already exists; delete it deliberately before rebuilding")
    OUT.parent.mkdir(parents=True, exist_ok=True)

    files = [p for p in tracked_files() if keep(p)]
    assert files, "nothing to archive"
    for p in files:
        assert not p.startswith("paper/"), f"paper/ leaked in: {p}"
        assert not p.startswith("SUBMISSIONS/"), f"SUBMISSIONS/ leaked in: {p}"
        assert not p.startswith(".results/"), f".results/ leaked in: {p}"

    readded = []
    for src_prefix, dst_prefix in READD.items():
        for p in tracked_files():
            if p.startswith(src_prefix) and Path(p).suffix not in EXCLUDED_SUFFIXES:
                readded.append((p, dst_prefix + p[len(src_prefix):]))
    assert readded, "the re-add list matched nothing"

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            src = ROOT / p
            if src.is_file():
                z.write(src, p)
        for src_rel, arc in readded:
            src = ROOT / src_rel
            if src.is_file():
                z.write(src, arc)

    size = OUT.stat().st_size / 1e6
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  {len(files) + len(readded)} files, {size:.1f} MB")
    print(f"  re-added {len(readded)} files from paper/analyses/ as analyses/")
    print(f"  excluded prefixes: {', '.join(EXCLUDED_DIRS)}")


if __name__ == "__main__":
    main()
