"""LLM agent with feedback: each round the model sees the unevaluated candidates and the history of
(candidate, e_hull) results it chose, and returns k ids with a one-line reason each (strict JSON).

Every prompt/response is cached under data/cache/llm/ keyed by (model, seed, round, prompt hash), so replays are
free and reproducible. Unknown/duplicate ids are dropped; the campaign fills the gap randomly (and logs it).
"""
from __future__ import annotations

import json
import re
import time

from ..utils import CACHE, dump_json, get_logger, load_config, load_json, optional_env, require_env, text_hash
from .base import Strategy

log = get_logger("llm_agent")
LLM_DIR = CACHE / "llm"

# Original prompt from the project brief (kept as the "llm_basic" ablation arm).
SYSTEM_BASIC = (
    "You are an expert solid-state chemist running a computational screening campaign for lithium-ion conducting "
    "halide solid electrolytes (Li3MX6 and Li2MX6 families, e.g. Li3YCl6, Li3InCl6, Li2ZrCl6). Each evaluation is an "
    "expensive simulation that returns the energy above the convex hull (e_hull, eV/atom) of the candidate, computed "
    "with a universal ML interatomic potential against competing phases. Lower is more stable; a candidate with "
    "e_hull <= {thr} eV/atom counts as a stable hit. Your goal is to find as many distinct stable compositions as "
    "possible with a fixed budget. Use chemical reasoning: ionic radii and radius ratios, oxidation-state stability, "
    "known structure-type preferences (trigonal vs monoclinic), halide size, and above all the results so far "
    "(exploit analogues of hits, avoid analogues of clear failures, but keep exploring untested chemistries). "
    "Do not pick several candidates with the same composition unless you have a reason."
)

# Main agent: same task, with the scoring rule and the energy scale stated explicitly, and richer feedback
# (competing phases of every evaluated candidate).
SYSTEM = SYSTEM_BASIC.replace(
    "Do not pick several candidates with the same composition unless you have a reason.",
    "SCORING: you earn one point per DISTINCT composition (reduced formula) that you find at e_hull <= {thr} eV/atom. "
    "Several candidates can share a composition (the same formula in the trigonal Li3YCl6-type and the monoclinic "
    "Li3InCl6-type cell): evaluating a second one only pays if the first was not a hit, so within one round pick at "
    "most one candidate per composition. Compositions you have already found are removed from your list. Whether a "
    "compound is already known from the literature does not matter for the score. "
    "ENERGY SCALE: e_hull values in this family are small, typically -0.07 to +0.08 eV/atom, so outcomes are "
    "decided by trends of a few meV/atom; after each round, update on the measured results (which metals, halides, "
    "prototypes and mixtures land below the hull, and which competing phases they decompose into) rather than on "
    "prior expectations."
)

USER = """Round {round} of {n_rounds}. Evaluations used: {used}/{budget}.

RESULTS SO FAR (id | description | e_hull eV/atom | hit{dec_header}):
{history}

UNEVALUATED CANDIDATES (id | description):
{pool}

Choose exactly {k} candidate ids from the UNEVALUATED list to evaluate next.
Return strict JSON only, in this exact form:
{{"picks": [{{"id": "<id>", "reason": "<one line of chemical reasoning>"}}]}}"""


class LLMAgentStrategy(Strategy):
    name = "llm"
    system_template = SYSTEM
    feedback_decomposition = True

    def __init__(self, seed, evaluator, batch_size=10, budget=150):
        super().__init__(seed, evaluator, batch_size)
        cfg = load_config()
        self.cfg = cfg["llm"]
        self.thr = cfg["labels"]["hit_threshold"]
        self.budget = budget
        self.model = self._model_name()
        self.base_url = optional_env("LLM_BASE_URL", self.cfg["default_base_url"])
        self._client = None
        self.calls: list[dict] = []  # per round metadata (cached or live, n_valid, n_filled, ...)

    def _model_name(self) -> str:
        return optional_env("LLM_MODEL", self.cfg["default_model"])

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=require_env("LLM_API_KEY"), base_url=self.base_url,
                                  timeout=self.cfg["timeout_s"], max_retries=2)
        return self._client

    def _prompt(self, pool_ids, history, k):
        hist_lines = []
        for cid, e in history:
            if e is None:
                hist_lines.append(f"{cid} | {self.ev.describe(cid)} | failed | no")
            else:
                line = f"{cid} | {self.ev.describe(cid)} | {e:+.3f} | {'yes' if e <= self.thr else 'no'}"
                if self.feedback_decomposition:
                    phases = " + ".join(self.ev.decomposition(cid))
                    line += f" | {'below' if e <= 0 else 'decomposes to'} {phases}"
                hist_lines.append(line)
        pool = sorted(pool_ids)
        import random

        random.Random(f"llm-shuffle-{self.seed}-{self.round}").shuffle(pool)
        pool_lines = [f"{cid} | {self.ev.describe(cid)}" for cid in pool]
        n_rounds = -(-self.budget // self.batch_size)
        return USER.format(dec_header=" | competing phases" if self.feedback_decomposition else "",
                           round=self.round + 1, n_rounds=n_rounds, used=len(history), budget=self.budget,
                           history="\n".join(hist_lines) if hist_lines else "(none yet: this is the first round)",
                           pool="\n".join(pool_lines), k=k)

    def _call(self, system, user) -> tuple[str, dict]:
        params = {"temperature": self.cfg["temperature"], "response_format": {"type": "json_object"}}
        last_err = None
        for attempt in range(self.cfg["max_retries"] + 2):
            try:
                kwargs = dict(model=self.model, messages=[{"role": "system", "content": system},
                                                           {"role": "user", "content": user}], **params)
                resp = self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or "", {"params": {k: v for k, v in params.items()},
                                                                "usage": getattr(resp, "usage", None) and resp.usage.model_dump()}
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                last_err = msg
                if "temperature" in msg and "temperature" in params:
                    log.warning("model rejects temperature; retrying with the default")
                    params.pop("temperature")
                    continue
                if "response_format" in msg and "response_format" in params:
                    log.warning("model rejects response_format; retrying without")
                    params.pop("response_format")
                    continue
                log.warning("LLM call failed (%s); retry %d", msg[:200], attempt + 1)
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"LLM call failed: {last_err}")

    @staticmethod
    def _parse(text: str) -> list[dict]:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, re.S)
            data = json.loads(m.group(0)) if m else {}
        picks = data.get("picks", []) if isinstance(data, dict) else []
        return [p for p in picks if isinstance(p, dict) and "id" in p]

    def propose(self, pool_ids, history, k):
        system = self.system_template.format(thr=self.thr)
        user = self._prompt(pool_ids, history, k)
        key = (f"{self.name}_{re.sub(r'[^A-Za-z0-9.-]', '_', self.model)}_seed{self.seed}_round{self.round:02d}_"
               f"{text_hash(system + user)}")
        path = LLM_DIR / f"{key}.json"
        rec = load_json(path)
        cached = rec is not None
        if not cached:
            t0 = time.time()
            text, meta = self._call(system, user)
            rec = {"model": self.model, "base_url": self.base_url, "seed": self.seed, "round": self.round,
                   "system": system, "prompt": user, "response": text, "seconds": time.time() - t0, **meta}
            dump_json(rec, path)
        picks = self._parse(rec["response"])
        pool = set(pool_ids)
        valid, seen = [], set()
        for p in picks:
            cid = str(p["id"]).strip()
            if cid in pool and cid not in seen:
                seen.add(cid)
                valid.append(cid)
                self.reasons[cid] = str(p.get("reason", "")).strip()
        self.calls.append({"round": self.round, "cached": cached, "n_returned": len(picks), "n_valid": len(valid[:k])})
        self.round += 1
        return valid[:k]


class LLMBasicStrategy(LLMAgentStrategy):
    """Ablation: the brief's original prompt and e_hull-only feedback."""
    name = "llm_basic"
    system_template = SYSTEM_BASIC
    feedback_decomposition = False


class LLMStrongStrategy(LLMAgentStrategy):
    """Same prompt and feedback as the main agent, with a stronger reasoning model (config llm.strong_model)."""
    name = "llm_strong"

    def _model_name(self) -> str:
        return self.cfg["strong_model"]


class LLMGPHybridStrategy(LLMStrongStrategy):
    """Hybrid: the GP surrogate (same features as GP-BO) shortlists the 30 candidates with the highest posterior
    P(hit); the reasoning LLM sees that shortlist with the GP probabilities plus the full history, and picks k with
    a chemical reason. Round 1 (no data for the GP yet) is the plain LLM agent."""
    name = "llm_gp"
    shortlist = 30
    system_template = SYSTEM + (
        " From round 2 on, a Gaussian-process surrogate trained on your results pre-selects the {n} most promising "
        "unevaluated candidates and reports its probability that each is a hit; combine it with chemical reasoning "
        "(the surrogate knows only composition statistics, you know chemistry)."
    ).replace("{n}", "30")

    def __init__(self, seed, evaluator, batch_size=10, budget=150):
        super().__init__(seed, evaluator, batch_size, budget)
        from .gp_bo import GPBOStrategy

        self.gp = GPBOStrategy(seed, evaluator, batch_size)
        self._p: dict[str, float] = {}

    def propose(self, pool_ids, history, k):
        pool = sorted(pool_ids)
        score = self.gp.scores(pool, history)
        self._p = {}
        if score is not None:
            order = sorted(range(len(pool)), key=lambda i: -score[i])[: self.shortlist]
            self._p = {pool[i]: float(score[i]) for i in order}
            pool = [pool[i] for i in order]
        return super().propose(pool, history, k)

    def _prompt(self, pool_ids, history, k):
        text = super()._prompt(pool_ids, history, k)
        if not self._p:
            return text
        lines = []
        for line in text.splitlines():
            cid = line.split(" | ", 1)[0]
            if cid in self._p and " | " in line and cid in pool_ids:
                line = f"{line} | GP P(hit)={self._p[cid]:.2f}"
            lines.append(line)
        return "\n".join(lines).replace("UNEVALUATED CANDIDATES (id | description):",
                                        "UNEVALUATED CANDIDATES, GP SHORTLIST (id | description | GP P(hit)):")
