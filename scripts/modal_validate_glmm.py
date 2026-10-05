"""Run the GLMM validation units on Modal, one container each.

Thin orchestration: every number comes from `paper/analyses/validate_glmm.py`, which runs
the same way locally. A unit writes its shard to the volume and commits the moment it
finishes, so a container that dies costs that unit and nothing else, and a rerun skips
whatever the volume already holds.

The image carries R and `lme4`, which the laptop's R cannot fit at this size: whether
`glmer` completes here is itself one of the answers this run is for.

    modal run --detach scripts/modal_validate_glmm.py
    modal run --detach scripts/modal_validate_glmm.py --units reduced_design,h6_shape
    modal run scripts/modal_validate_glmm.py::collect      # assemble what is done so far
"""

import os

import modal

ANALYSES = os.path.join(os.path.expanduser("~"),
                        "Documents/GitHub/ecological-discrepancy-covid/paper/analyses")
# `paths.py` derives the project root from where it sits, so the container mirrors the
# repository's shape rather than a flat directory.
REMOTE = "/root/repo/paper/analyses"

image = (
    modal.Image.debian_slim(python_version="3.13")
    # Debian's prebuilt r-cran-* packages: lme4 from source needs a compiler chain and
    # nlopt, and a silent failure there would look like "glmer does not work here".
    .apt_install("r-base-core", "r-cran-lme4", "r-cran-jsonlite", "r-cran-matrix")
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
    )
    .env({"PYTHONPATH": REMOTE})
    .add_local_file(os.path.join(ANALYSES, "glmm.py"), "/root/repo/paper/analyses/glmm.py", copy=True)
    .add_local_file(os.path.join(ANALYSES, "validate_glmm.py"),
                    "/root/repo/paper/analyses/validate_glmm.py", copy=True)
    .add_local_file(os.path.join(ANALYSES, "paths.py"), "/root/repo/paper/analyses/paths.py", copy=True)
    .add_local_file(os.path.join(ANALYSES, "r/glmer_reference.R"),
                    "/root/repo/paper/analyses/r/glmer_reference.R", copy=True)
    .add_local_file(os.path.join(ANALYSES, "results/mexico_cells.csv"),
                    "/root/repo/paper/analyses/results/mexico_cells.csv", copy=True)
    .add_local_file(os.path.join(ANALYSES, "results/mexico_glmm_fits.json"),
                    "/root/repo/paper/analyses/results/mexico_glmm_fits.json", copy=True)
)

app = modal.App("jamia-glmm-validation", image=image)
volume = modal.Volume.from_name("jamia-glmm-validation", create_if_missing=True)

SHARDS = "/results/shards"
TIMEOUT = 86_400


@app.function(cpu=2.0, memory=16_384, timeout=TIMEOUT, volumes={"/results": volume})
def run_one(name: str) -> str:
    """One validation unit, checkpointed to the volume the moment it finishes."""
    import time
    from datetime import datetime, timezone

    import validate_glmm

    volume.reload()
    done = validate_glmm.load_shard(name, SHARDS)
    if done is not None:
        print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {name}: already on the volume")
        return name

    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {name}: starting", flush=True)
    started = time.time()
    payload = validate_glmm.run_unit(name)
    validate_glmm.save_shard(name, payload, SHARDS, time.time() - started)
    volume.commit()
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {name}: done in "
          f"{time.time() - started:.0f}s", flush=True)
    return name


@app.function(image=image, timeout=3600, volumes={"/results": volume})
def assemble() -> dict:
    """Read every shard the volume holds and return the assembled result."""
    from datetime import datetime

    import validate_glmm

    volume.reload()
    payloads = {}
    for name in validate_glmm.unit_names():
        done = validate_glmm.load_shard(name, SHARDS)
        if done is not None:
            payloads[name] = done
    return validate_glmm.assemble(
        payloads, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def _stamp(result):
    """Provenance comes from the machine that holds the repository, not the container.

    The image carries no git, so the commit an assembled result names is recorded here,
    beside the shards it was built from.
    """
    import validate_glmm
    from paths import run_metadata

    result["run"] = run_metadata(result["run"]["run_id"])
    return result


@app.function(image=image, timeout=600)
def r_probe() -> dict:
    """Does this image have a working lme4 at all, and which versions?"""
    import subprocess

    script = ("cat(R.version.string, '|', as.character(packageVersion('lme4')), '|', "
              "as.character(packageVersion('Matrix')))")
    attempt = subprocess.run(["Rscript", "-e", script], capture_output=True, text=True,
                             check=False)
    return {"returncode": attempt.returncode, "stdout": attempt.stdout.strip(),
            "stderr": attempt.stderr.strip()[-400:]}


@app.local_entrypoint()
def main(units: str = "", probe: bool = False):
    import json
    from pathlib import Path

    import validate_glmm

    if probe:
        print(json.dumps(r_probe.remote(), indent=2))
        return

    names = units.split(",") if units else validate_glmm.unit_names()
    print(f"dispatching {len(names)} units")
    # One unit that raises must not cancel the other fifty-five: the exception comes back
    # as a value, is named, and the rest keep going.
    broken = []
    for name, outcome in zip(names, run_one.map(names, return_exceptions=True)):
        if isinstance(outcome, Exception):
            broken.append((name, f"{type(outcome).__name__}: {outcome}"))
            print(f"  FAILED {name}: {type(outcome).__name__}: {outcome}", flush=True)
        else:
            print(f"  finished {outcome}", flush=True)

    result = _stamp(assemble.remote())
    result["units_that_raised"] = dict(broken)
    destination = Path(validate_glmm.OUTPUT)
    saved = validate_glmm.write_result(result, destination)
    print(f"\nassembled {len(result['units_completed'])} of "
          f"{len(result['units_expected'])} units into {saved}")
    if broken:
        print(f"{len(broken)} units raised: {[name for name, _ in broken]}")


@app.local_entrypoint()
def collect():
    """Write the result file from whatever the volume already holds."""
    import validate_glmm

    result = _stamp(assemble.remote())
    saved = validate_glmm.write_result(result, validate_glmm.OUTPUT)
    print(f"assembled {len(result['units_completed'])} of "
          f"{len(result['units_expected'])} units into {saved}")
