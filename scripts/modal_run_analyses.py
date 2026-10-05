"""Run the registered analyses on Modal, where `lme4` fits H6.

Thin orchestration: every number comes from the scripts in `paper/analyses`, unchanged.
The laptop's R segfaults on H6's cell table, so the pipeline runs in an environment whose
R does not, and the launcher - which is the machine that holds the repository - verifies
the tree is clean and passes the commit it verified.

The snapshot lives on its own volume and is never rebuilt: the loader hashes it and
refuses to return data unless it reproduces the primary analysis exactly, so running it
here is only safe if that check passes, which is what `verify` is for.

    modal run scripts/modal_run_analyses.py::tests           # the invariant suite
    modal run scripts/modal_run_analyses.py::verify          # the loader, and nothing else
    modal deploy scripts/modal_run_analyses.py               # then
    modal run scripts/modal_run_analyses.py::launch --stages s12

`launch` spawns on the deployed app, which is the only form that survives the laptop
sleeping: `--detach` keeps the last triggered function alive after the client goes away,
and a deployed function never depended on the client to begin with. `run` holds the
client open for the whole run and is for a stage short enough to watch.
"""

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone

import modal

ANALYSES = os.path.join(os.path.expanduser("~"),
                        "Documents/GitHub/ecological-discrepancy-covid/paper/analyses")
REMOTE = "/root/repo/paper/analyses"
REMOTE_TESTS = REMOTE + "/tests"
# Never copied, so never in the manifest the container checks itself against.
IGNORED = ["__pycache__", "results", "tests", "logs", "attestations", ".DS_Store"]
VALIDATION = "/root/repo/paper/analyses/validation/glmm_validation.json"

image = (
    modal.Image.debian_slim(python_version="3.13")
    # git: not for the run, which is handed a verified commit, but so the suite can
    # build a repository and check that the clean-tree guard refuses a dirty one.
    .apt_install("git", "r-base-core", "r-cran-lme4", "r-cran-jsonlite", "r-cran-matrix")
    .run_commands(
        "R -e \"stopifnot(requireNamespace('lme4'), requireNamespace('jsonlite')); "
        "cat(R.version.string, as.character(packageVersion('lme4')))\"",
    )
    .pip_install(
        "numpy==2.5.1",
        "scipy==1.18.0",
        "pandas==3.0.3",
        "patsy==1.0.2",
        "statsmodels==0.14.6",
        # the primary analysis the loader checks itself against imports it
        "tqdm==4.70.1",
    )
    # The suite runs here rather than on the laptop: it fits models on tens of thousands
    # of records, and running it locally exhausted the machine's memory and swap.
    .pip_install("pytest==9.1.1")
    .env({"PYTHONPATH": REMOTE})
    .add_local_dir(ANALYSES, REMOTE, copy=True, ignore=IGNORED)
    # Copied separately, so the tests are available here without entering the manifest
    # that fixes which code produced a number.
    .add_local_dir(os.path.join(ANALYSES, "tests"), REMOTE_TESTS, copy=True,
                   ignore=["__pycache__", ".DS_Store"])
    # The results directory is a mounted volume in the container and does not carry the
    # validation artifact, so it travels separately and the preflight is pointed at it.
    .add_local_file(os.path.join(ANALYSES, "results/glmm_validation.json"),
                    VALIDATION, copy=True)
)

app = modal.App("jamia-analyses", image=image)
snapshot = modal.Volume.from_name("jamia-mexico-snapshot")
results = modal.Volume.from_name("jamia-analysis-results", create_if_missing=True)

DATA = "/root/repo/data/mexico_covid"
OUT = "/root/repo/paper/analyses/results"
LOGS = os.path.join(ANALYSES, "logs")
STAGES = ("s11", "s12")
TIMEOUT = 86_400


def _manifest(root):
    """sha256 of every file that will be copied, keyed by its path inside the image."""
    import hashlib
    from pathlib import Path

    entries = {}
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file() or any(part in IGNORED for part in path.parts):
            continue
        entries[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return entries


def _prepare(commit, manifest):
    """Prove the copied files are the commit the launcher verified, then let it run.

    Being told a commit is not the same as holding it. The launcher hashes what it is
    about to send; the container hashes what arrived and refuses to run on a difference,
    which closes the gap between "the launcher verified X" and "this executed X".
    """
    import hashlib
    import os
    from pathlib import Path

    mismatched = []
    for relative, expected in sorted(manifest.items()):
        arrived = Path(REMOTE) / relative
        if not arrived.exists():
            mismatched.append(f"{relative}: missing")
        elif hashlib.sha256(arrived.read_bytes()).hexdigest() != expected:
            mismatched.append(f"{relative}: differs")
    if mismatched:
        raise RuntimeError("the container does not hold the verified commit: "
                           + "; ".join(mismatched))

    os.environ["ANALYSIS_COMMIT"] = commit
    os.environ["VALIDATION_ARTIFACT"] = VALIDATION
    Path(OUT).mkdir(parents=True, exist_ok=True)
    print(f"verified {len(manifest)} files against the launcher's manifest", flush=True)


@app.function(cpu=4.0, memory=32_768, timeout=TIMEOUT,
              volumes={DATA: snapshot, OUT: results})
def verify_loader(commit: str, manifest: dict) -> dict:
    """Does the snapshot reproduce the primary analysis in this environment?"""
    import json

    _prepare(commit, manifest)
    import mexico_confirmed_cases

    frame = mexico_confirmed_cases.load()
    results.commit()
    audit = json.load(open(f"{OUT}/mexico_loader_audit.json"))
    return {
        "records": int(len(frame)),
        "deaths": int(frame["died"].sum()),
        "treating_states": int(frame["site"].nunique()),
        "counts": audit.get("counts"),
        "outside_catalogue": audit.get("columns", {}).get("outside_catalogue"),
    }


@app.function(cpu=8.0, memory=65_536, timeout=TIMEOUT,
              volumes={DATA: snapshot, OUT: results})
def run_stage(stage: str, commit: str, manifest: dict) -> str:
    """One registered analysis, writing its result file to the results volume."""
    from datetime import datetime, timezone

    _prepare(commit, manifest)
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {stage} starting", flush=True)
    if stage == "s11":
        import s11_site_definitions as analysis
    elif stage == "s12":
        import s12_composition_ladder as analysis
        # Writing a checkpoint to the mounted path is not the volume holding it. Flushing
        # each one as it is written is what makes a kill cost the unit in progress rather
        # than everything since the last background commit.
        analysis.checkpoint_written = results.commit
    else:
        raise ValueError(f"no stage named {stage}")
    try:
        analysis.main()
    finally:
        try:
            results.commit()
        except Exception as error:   # the run's own failure is the one worth reporting
            print(f"  the closing volume commit failed: {error}", flush=True)
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {stage} done", flush=True)
    return stage


@app.function(timeout=600, volumes={OUT: results})
def write_then_raise(marker: str, flush: bool):
    """Write a checkpoint and fail, so persistence can be observed instead of assumed."""
    from pathlib import Path

    path = Path(OUT) / "smoke" / f"{marker}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"marker": marker, "flushed": flush}))
    if flush:
        results.commit()
    raise RuntimeError(f"intentional failure after writing {path}")


@app.function(timeout=600, volumes={OUT: results})
def read_markers(markers: list) -> dict:
    """What a container that starts afterwards finds on the volume."""
    from pathlib import Path

    results.reload()
    return {marker: (json.loads(found.read_text())
                     if (found := Path(OUT) / "smoke" / f"{marker}.json").exists()
                     else None)
            for marker in markers}


def _verified_commit():
    """The launcher does the check the container cannot: a clean, committed tree.

    `paths` is loaded by file rather than by name: this runs on the laptop, where the
    analyses are not on the import path, and the module is loaded here rather than at
    import so that the container, which has no such file, can still import this script.
    """
    import importlib.util

    specification = importlib.util.spec_from_file_location(
        "paths", os.path.join(ANALYSES, "paths.py"))
    paths = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(paths)

    paths.require_clean_tree()
    return paths._git("rev-parse", "HEAD").stdout.strip()


@app.local_entrypoint()
def verify():
    import json

    print(json.dumps(verify_loader.remote(_verified_commit(), _manifest(ANALYSES)),
                     indent=2))


@app.function(cpu=8.0, memory=32_768, timeout=3_600, volumes={DATA: snapshot})
def run_tests(selection: str) -> int:
    """The invariant suite, in the image the analyses themselves run in.

    The snapshot volume is mounted for the module beside the data that
    `mexico_confirmed_cases` executes at import to check itself against, not for the
    1.9 GB CSV, which no test reads.
    """
    arguments = ["python", "-m", "pytest", REMOTE_TESTS, "-q"]
    if selection:
        arguments += ["-k", selection]
    return subprocess.run(arguments, cwd=REMOTE, check=False).returncode


@app.local_entrypoint()
def tests(k: str = ""):
    """Run the suite remotely. It fits models, so it does not run on the laptop."""
    code = run_tests.remote(k)
    print("the suite passed" if code == 0 else f"pytest exited {code}")


@app.local_entrypoint()
def smoke():
    """Does a checkpoint written before a crash survive into the next container?

    The claim that it does was read off the Modal client's source rather than measured,
    and a durability claim nobody has observed is the kind the next failed run disproves.
    Both arms raise after writing; only one flushes the volume first.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    markers = {True: f"flushed_{stamp}", False: f"unflushed_{stamp}"}
    for flush, marker in markers.items():
        try:
            write_then_raise.remote(marker, flush)
        except Exception as error:
            print(f"  {marker}: raised as intended ({type(error).__name__})")
    found = read_markers.remote(list(markers.values()))
    for marker, record in found.items():
        print(f"  {marker}: {'survived' if record else 'LOST'}")
    print(json.dumps(found, indent=2))


@app.local_entrypoint()
def launch(stages: str = "s12"):
    """Spawn on the deployed app and return; the run does not depend on this client."""
    wanted = [stage.strip() for stage in stages.split(",") if stage.strip()]
    unknown = [stage for stage in wanted if stage not in STAGES]
    if not wanted:
        raise ValueError(f"no stage named in {stages!r}")
    if unknown:
        raise ValueError(f"no stage called {', '.join(unknown)}; known: {', '.join(STAGES)}")

    commit, manifest = _verified_commit(), _manifest(ANALYSES)
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    print(f"launching {', '.join(wanted)} from {commit[:12]}, "
          f"manifest {digest[:12]}, {len(manifest)} files")

    deployed = modal.Function.from_name(app.name, "run_stage")
    calls = {stage: deployed.spawn(stage, commit, manifest).object_id for stage in wanted}

    os.makedirs(LOGS, exist_ok=True)
    receipt = os.path.join(LOGS, f"launch_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json")
    with open(receipt, "w") as handle:
        json.dump({"spawned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "commit": commit, "manifest_sha256": digest,
                   "files_in_manifest": len(manifest), "calls": calls}, handle, indent=2)
    for stage, call in calls.items():
        print(f"  {stage}: {call}")
    print(f"receipt: {receipt}")


@app.local_entrypoint()
def run(stages: str = "s11,s12"):
    commit, manifest = _verified_commit(), _manifest(ANALYSES)
    print(f"running {stages} from {commit[:12]}, {len(manifest)} files in the manifest")
    for stage in run_stage.map(stages.split(","),
                               kwargs={"commit": commit, "manifest": manifest},
                               return_exceptions=True):
        print(f"  {stage}", flush=True)
