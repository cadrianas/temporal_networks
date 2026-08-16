"""
Smoke tests: every example script runs to completion.

The scripts in ``examples/`` are documentation. Each is referenced from
the README, and the two domain walkthroughs each back a tutorial page
under ``docs/``. They assert their own substantive claims — that the
snapshot pair straddling a gap is ``NaN``, that an intervention is
flagged as a change point while the resumption of data collection is
not, and so on — so running one to completion checks both that the
package still works and that the documented results still hold.

Without this, the examples are only linted, never executed, and can rot
unnoticed: they are not imported by any other test.

Each script runs in a fresh working directory because several of them
write plots and CSVs relative to the cwd, and nothing should land in the
repository.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"

# Every example script is expected to run to completion. Keep in sync
# with the directory; test_example_list_is_complete enforces that.
EXAMPLE_SCRIPTS = [
    "example_1_synthetic.py",
    "example_epidemic_abm.py",
    "example_full_integration.py",
    "example_multigap_report.py",
    "example_social_contacts.py",
    "example_specs_01_02_03.py",
]


def _run_example(script: str, cwd: Path) -> subprocess.CompletedProcess:
    """Run one example script in ``cwd`` and return the completed process."""
    env = dict(os.environ)
    # Import this checkout even where the editable install is not picked
    # up, and never attempt to open a plotting window.
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if part
    )
    env["MPLBACKEND"] = "Agg"

    return subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / script)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(cwd),
        timeout=600,
    )


def test_example_list_is_complete():
    """A new example must be added to EXAMPLE_SCRIPTS, not silently skipped."""
    on_disk = sorted(path.name for path in EXAMPLES_DIR.glob("example_*.py"))
    assert on_disk == sorted(EXAMPLE_SCRIPTS), (
        "examples/ and EXAMPLE_SCRIPTS have diverged; add the new example "
        "to the list so it is covered by these smoke tests"
    )


@pytest.mark.parametrize("script", EXAMPLE_SCRIPTS)
def test_example_runs_to_completion(script, tmp_path):
    """The script exits 0, so all of its internal assertions held."""
    result = _run_example(script, tmp_path)

    assert result.returncode == 0, (
        f"{script} exited with {result.returncode}\n"
        f"--- stdout (tail) ---\n{result.stdout[-2000:]}\n"
        f"--- stderr (tail) ---\n{result.stderr[-2000:]}"
    )


@pytest.mark.parametrize(
    "script", ["example_social_contacts.py", "example_epidemic_abm.py"]
)
def test_domain_walkthroughs_are_reproducible(script, tmp_path):
    """
    The two tutorial scripts produce identical output on every run.

    Their documentation pages quote concrete numbers, which is only
    honest if the scripts are deterministic. Community detection is
    randomised, so example_social_contacts.py seeds igraph's generator
    explicitly; this test is what stops that seeding from being dropped.
    """
    first_dir, second_dir = tmp_path / "a", tmp_path / "b"
    first_dir.mkdir()
    second_dir.mkdir()

    first = _run_example(script, first_dir)
    second = _run_example(script, second_dir)

    assert first.returncode == 0 and second.returncode == 0
    assert first.stdout == second.stdout, (
        f"{script} is not reproducible: two runs produced different "
        f"output. Check that igraph's RNG is seeded with "
        f"ig.set_random_number_generator(random.Random(...))."
    )
