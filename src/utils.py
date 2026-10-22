"""Shared helpers: paths, config, secrets, JSON cache, structure hashing."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml

CODE_ROOT = Path(__file__).resolve().parent.parent
# On Modal the data lives on a mounted volume; locally it sits next to the code.
DATA_ROOT = Path(os.environ.get("OSI_DATA_ROOT", CODE_ROOT))
CACHE = DATA_ROOT / "data" / "cache"
RESULTS = DATA_ROOT / "results"
for _d in (CACHE, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def load_config() -> dict:
    with open(CODE_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def _load_dotenv() -> None:
    env = CODE_ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# Accept the standard OpenAI variable names as fallbacks (the Modal secret "openai-secret" uses them).
_FALLBACKS = {"LLM_API_KEY": "OPENAI_API_KEY", "LLM_BASE_URL": "OPENAI_BASE_URL", "LLM_MODEL": "OPENAI_MODEL"}


def require_env(name: str) -> str:
    _load_dotenv()
    val = os.environ.get(name) or os.environ.get(_FALLBACKS.get(name, ""), "")
    if not val:
        raise RuntimeError(f"Environment variable {name} is missing. Ask the human to set it in .env")
    return val


def optional_env(name: str, default: str | None = None) -> str | None:
    _load_dotenv()
    return os.environ.get(name) or os.environ.get(_FALLBACKS.get(name, ""), "") or default


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
    os.replace(tmp, path)


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def _json_default(o: Any) -> Any:
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "as_dict"):
        return o.as_dict()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def structure_hash(structure) -> str:
    """Hash of an input structure (species + rounded lattice/fractional coords), oxidation states ignored."""
    lat = [round(x, 4) for row in structure.lattice.matrix for x in row]
    sites = sorted(
        (site.specie.symbol if hasattr(site.specie, "symbol") else str(site.specie),
         tuple(round(c % 1.0, 4) % 1.0 for c in site.frac_coords))
        for site in structure
    )
    payload = json.dumps({"lat": lat, "sites": sites}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:20]


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def chemsys_key(elements) -> str:
    return "-".join(sorted(set(str(e) for e in elements)))
