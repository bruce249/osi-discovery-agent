"""Modal deployment: every compute step of the pipeline runs here, nothing heavy runs locally.

Usage (from the project root):
  modal run modal_app.py::smoke                         # API checks + CHGNet/MACE timing test
  modal run modal_app.py::step --name prototypes        # run any `python -m src.<name>` on Modal
  modal run --detach modal_app.py::relax_batch --model chgnet
  modal volume get osi-discovery-data results ./        # pull results back
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
APP_DIR = "/root/app"
VOL_DIR = "/vol"

app = modal.App("osi-discovery-agent")
vol = modal.Volume.from_name("osi-discovery-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(
        "pymatgen>=2024.1", "mp-api", "chgnet", "mace-torch", "ase>=3.23", "matminer",
        "scikit-learn", "pandas", "numpy", "matplotlib", "tqdm", "pyyaml", "marimo",
        "openai", "optimade[http_client]", "requests", "jarvis-tools",
    )
    .env({"OSI_DATA_ROOT": VOL_DIR, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2",
          "MKL_NUM_THREADS": "2", "MPLBACKEND": "Agg"})
    .add_local_dir(ROOT, APP_DIR, ignore=[".venv", ".venv/**", ".env", "data", "data/**", "results",
                                          "results/**", "**/__pycache__", "**/*.pyc", "__marimo__"])
)

# OpenAI key from the existing Modal secret; MP key (and optional LLM overrides) from the local .env
# (always exactly two secrets: the container must see the same dependency list as the local client)
_dotenv = modal.Secret.from_dotenv(ROOT) if (not modal.is_local() or (ROOT / ".env").exists()) else modal.Secret.from_dict({})
secrets = [modal.Secret.from_name("openai-secret"), _dotenv]
COMMON = dict(image=image, volumes={VOL_DIR: vol}, secrets=secrets)


def _setup():
    os.chdir(APP_DIR)
    if APP_DIR not in sys.path:
        sys.path.insert(0, APP_DIR)
    try:
        vol.reload()
    except Exception:  # noqa: BLE001 - reload fails if files are open; data is still readable
        pass


@app.function(**COMMON, cpu=4, memory=8192, timeout=3 * 3600)
def step(name: str, args: str = ""):
    """Run `python -m src.<name> <args>` inside the container (comma-separate names to chain), committing the
    volume after each."""
    _setup()
    for n in name.split(","):
        cmd = [sys.executable, "-m", f"src.{n}", *args.split()]
        print("running", " ".join(cmd), flush=True)
        rc = subprocess.run(cmd, cwd=APP_DIR).returncode
        vol.commit()
        if rc != 0:
            raise SystemExit(f"step {n} failed with exit code {rc}")


@app.function(**COMMON, cpu=2, memory=4096, timeout=1500, max_containers=80, retries=1)
def relax_task(task: dict) -> dict:
    _setup()
    from src.mlip import relax_structure

    res = relax_structure(task["structure"], task["model"], task["fmax"], task["steps"],
                          task["timeout_s"], task.get("mace_model", "medium"))
    res["task_hash"] = task["hash"]
    return res


@app.function(**COMMON, cpu=2, memory=8192, timeout=6 * 3600)
def relax_batch(model: str = "chgnet", tasks_file: str = ""):
    """Fan out every pending relaxation; write each result to the volume as it arrives."""
    _setup()
    from src.mlip import collect_tasks, pending, save_result
    from src.utils import load_json

    if tasks_file:
        path = Path(tasks_file) if tasks_file.startswith(VOL_DIR) else Path(VOL_DIR) / tasks_file  # relative to volume
        tasks = load_json(path)
        if tasks is None:
            raise FileNotFoundError(f'tasks file {path} not found on the volume')
    else:
        tasks = collect_tasks(model)
    todo = pending(tasks)
    # big cells first so the slowest jobs do not straggle at the end
    todo.sort(key=lambda t: -t.get("natoms", 0))
    by_hash = {t["hash"]: t for t in todo}
    print(f"[{model}] {len(tasks)} tasks, {len(todo)} pending", flush=True)
    t0, last_commit, n = time.time(), time.time(), 0
    for res in relax_task.map(todo, order_outputs=False, return_exceptions=True):
        if isinstance(res, BaseException):
            print("worker exception:", repr(res)[:300], flush=True)
            continue
        task = by_hash.get(res.get("task_hash"))
        if task is None:
            continue
        save_result(task, res)
        n += 1
        if n % 25 == 0 or time.time() - last_commit > 60:
            vol.commit()
            last_commit = time.time()
            print(f"[{model}] {n}/{len(todo)} done, {time.time() - t0:.0f}s", flush=True)
    vol.commit()
    print(f"[{model}] finished {n}/{len(todo)} in {time.time() - t0:.0f}s", flush=True)


@app.function(**COMMON, cpu=2, memory=4096, timeout=900)
def notebook_export():
    """Run the marimo notebook top to bottom from the cache and save a static HTML copy."""
    _setup()
    rc = subprocess.run(["marimo", "export", "html", "notebook.py", "-o", f"{VOL_DIR}/results/notebook.html"],
                        cwd=APP_DIR).returncode
    vol.commit()
    if rc != 0:
        raise SystemExit(f"marimo export failed with exit code {rc}")


@app.function(**COMMON, cpu=4, memory=8192, timeout=24 * 3600, max_containers=1, scaledown_window=900)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=2718, startup_timeout=180)
def notebook():
    """Public, interactive notebook (marimo app mode: visitors use the controls, they cannot edit code).
    Deploy: modal deploy modal_app.py"""
    _setup()
    subprocess.Popen(["marimo", "run", "notebook.py", "--host", "0.0.0.0", "--port", "2718", "--headless"],
                     cwd=APP_DIR)


@app.function(**COMMON, cpu=1, memory=1024, timeout=300)
def env_check():
    """Print which expected variables are present (names only, never values)."""
    for k in ("MP_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY", "LLM_BASE_URL", "OPENAI_BASE_URL", "LLM_MODEL", "OPENAI_MODEL"):
        print(f"{k}: {'set' if os.environ.get(k) else 'missing'}")
    print("other OPENAI*/LLM* names:", sorted(k for k in os.environ if k.startswith(("OPENAI", "LLM"))))


@app.function(**COMMON, cpu=2, memory=4096, timeout=1800)
def smoke():
    """Verify library APIs and time CHGNet/MACE relaxations on a 32-atom chloride (5 runs each)."""
    _setup()
    import inspect
    import json

    import numpy as np
    from pymatgen.core import Lattice, Structure

    from src.mlip import relax_structure
    from src.utils import RESULTS, dump_json

    report = {}
    import ase, chgnet, mace, pymatgen.core, torch  # noqa: E401

    report["versions"] = {"ase": ase.__version__, "chgnet": chgnet.__version__, "torch": torch.__version__,
                          "pymatgen": pymatgen.core.__version__,
                          "mace": getattr(mace, "__version__", "?")}
    from pymatgen.analysis.structure_prediction.substitution_probability import SubstitutionProbability
    from pymatgen.core import Species

    sp = SubstitutionProbability()
    report["subprob_sig"] = str(inspect.signature(sp.prob))
    report["subprob_Y_Er"] = sp.prob(Species("Y", 3), Species("Er", 3))
    report["subprob_Cl_Br"] = sp.prob(Species("Cl", -1), Species("Br", -1))
    report["subprob_Y_Y"] = sp.prob(Species("Y", 3), Species("Y", 3))
    from mp_api.client import MPRester

    report["mp_summary_search_params"] = [p for p in inspect.signature(MPRester().materials.summary.search).parameters][:60] if os.environ.get("MP_API_KEY") else "no key"

    rng = np.random.default_rng(0)
    base = Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.14), ["Li", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    base.make_supercell([1, 2, 2])  # 32 atoms
    timings = {}
    for model, fmax in (("chgnet", 0.1), ("mace", 0.05)):
        ts = []
        for i in range(5 if model == "chgnet" else 3):
            s = base.copy()
            s.perturb(0.15)
            s.apply_strain(rng.uniform(-0.04, 0.04))
            r = relax_structure(s.as_dict(), model, fmax, 300, 900)
            ts.append({k: r.get(k) for k in ("ok", "seconds", "steps", "converged", "e_per_atom", "error")})
        timings[model] = ts
    report["timings"] = timings
    report["cpu_count"] = os.cpu_count()
    dump_json(report, RESULTS / "smoke.json")
    vol.commit()
    print(json.dumps(report, indent=1, default=str))
