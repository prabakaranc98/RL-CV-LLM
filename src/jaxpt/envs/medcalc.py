"""MedCalc-Bench as an RLVR task: data, prompts, answer extraction and the reward checker.

Evaluation (E0) and training (X1) both import this module, so the model is always scored the same way.

Data: MedCalc-Bench Verified v1.0.8 (nsk7153/MedCalc-Bench-Verified, CC-BY-SA-4.0), the original authors'
re-audited successor of v1.2: 10,538 train / 1,100 test / 55 one-shot (one worked example per calculator).
The original v1.2 (ncbi/MedCalc-Bench) stays loadable with source="v1.2", for comparison.
Each row has a patient note, a question, a ground-truth answer and an accepted [lower, upper] range: about +-5% for
decimals, exact for integers and dates. Gestational age answers are (weeks, days) pairs, compared as total days.
Known bad label in v1.2: train-3175 (MDRD GFR 4.7e-07).

Reward contract (vanilla RLVR):
  * the response must contain exactly one distinct \\boxed{...} answer, holding exactly one number (or one date);
  * reward 1.0 if that value is inside the accepted range, else 0.0 (an invalid format also scores 0.0).
Single value only: a box such as "68-76" is rejected, so a range cannot straddle the accepted window.
"""
import ast
import re
from dataclasses import dataclass
from datetime import datetime

SOURCES = {   # name -> (Hugging Face repo, pinned revision, parquet path)
    "verified": ("nsk7153/MedCalc-Bench-Verified", "58fc3de568bfc53a68f32327e58f60dc6eaa17f8",   # release v1.0.8
                 "data/{split}-00000-of-00001.parquet"),
    "v1.2": ("ncbi/MedCalc-Bench", "refs/convert/parquet", "default/{split}/0000.parquet"),
}
HEADER = "Solve the clinical calculation. Reason step by step, then give the final answer in \\boxed{}.\n\n"
STOP = ["\n\nPatient note:", "\nPatient note:", "\n\nPatient values:", "\nPatient values:"]       # a base model tends to continue with a new problem

_BOX = re.compile(r"\\boxed\{([^{}]*)\}")
_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")      # allows 4.7e-07
_WEEKS = re.compile(r"(\d+)\s*weeks?"); _DAYS = re.compile(r"(\d+)\s*days?")


@dataclass(frozen=True)
class Example:
    id: str
    calculator: str
    category: str
    output_type: str        # "decimal" | "integer" | "date"
    note_type: str          # "Extracted" (real case report) | "Synthetic" | "Template"
    note: str
    question: str
    answer: str
    lower: str
    upper: str
    explanation: str
    entities: str           # 'Relevant Entities': the gold values a clinician would extract from the note


def load(split, source="verified"):
    """Load "train", "test" (or "one_shot", Verified only) from a pinned Hub revision. Returns a list of Example."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    repo, revision, pattern = SOURCES[source]
    path = hf_hub_download(repo, pattern.format(split=split), repo_type="dataset", revision=revision)
    rows = pq.read_table(path).to_pylist()
    return [Example(id=f"{split}-{r['Row Number']}", calculator=r["Calculator Name"], category=r["Category"],
                    output_type=r["Output Type"], note_type=r["Note Type"], note=r["Patient Note"], question=r["Question"],
                    answer=str(r["Ground Truth Answer"]), lower=str(r["Lower Limit"]), upper=str(r["Upper Limit"]),
                    explanation=r["Ground Truth Explanation"],
                    entities=r["Relevant Entities"]) for r in rows]


# ------------------------------------------------------------------------------------------- prompts
def values_text(ex):
    """The gold 'Relevant Entities' as plain lines, e.g. 'age: 87 years'. Used to remove extraction from the task."""
    ents = ast.literal_eval(ex.entities)
    lines = []
    for k, v in ents.items():
        if isinstance(v, (list, tuple)) and len(v) == 2 and isinstance(v[1], str):
            lines.append(f"{k}: {v[0]} {v[1]}")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)


def problem_block(ex, source="note"):
    """source='note': the model reads the patient note; source='values': it gets the gold extracted values instead."""
    given = f"Patient note: {ex.note}" if source == "note" else f"Patient values:\n{values_text(ex)}"
    return f"{given}\nQuestion: {ex.question}\nSolution:"


def worked_block(ex, source="note"):
    return f"{problem_block(ex, source)} {ex.explanation.strip()} The answer is \\boxed{{{ex.answer}}}.\n\n"


def default_shot(train):
    """The fixed one-shot example: the shortest real (Extracted) mean-arterial-pressure case in train."""
    pool = [e for e in train if e.calculator.startswith("Mean Arterial Pressure") and e.note_type == "Extracted"]
    return min(pool, key=lambda e: len(e.note))


def per_calculator_shots(one_shot):
    """The benchmark's own one-shot protocol: one worked example per calculator (the Verified 'one_shot' split)."""
    return {e.calculator: e for e in one_shot}


def build_prompt(ex, shot, source="note"):
    """Header, one worked example, then the problem; the model continues after 'Solution:'.
    The worked example uses the same source (note or values) as the problem, so the format is consistent."""
    return HEADER + worked_block(shot, source) + problem_block(ex, source)


def prompter(shots="generic", source="note", data="verified"):
    """ex -> prompt, for the two one-shot protocols of E0: "generic" (one fixed MAP example for every problem)
    or "per-calculator" (the benchmark's own worked example for the problem's calculator)."""
    if shots == "per-calculator":
        by_calc = per_calculator_shots(load("one_shot", "verified"))
        return lambda ex: build_prompt(ex, by_calc[ex.calculator], source)
    generic = default_shot(load("train", data))
    return lambda ex: build_prompt(ex, generic, source)


# ------------------------------------------------------------------------------------------- checker
def extract(text):
    """The single distinct \\boxed{} content, or None (zero boxes, or boxes that disagree)."""
    boxes = {b.strip() for b in _BOX.findall(text)}
    return boxes.pop() if len(boxes) == 1 else None


def _date(s):
    try:
        return datetime.strptime(s.strip(), "%m/%d/%Y").date()
    except ValueError:
        return None


def _weeks_days(s):
    """'(16 weeks, 1 days)' or "('16 weeks', '1 days')" -> total days; None if not that shape."""
    w, d = _WEEKS.search(s), _DAYS.search(s)
    return 7 * int(w.group(1)) + int(d.group(1)) if w and d else None


def reward(text, ex):
    """Score one response. Returns (reward, info) with info = {valid, parsed, reason}."""
    boxed = extract(text)
    if boxed is None:
        return 0.0, {"valid": False, "parsed": None, "reason": "no single boxed answer"}
    if "week" in ex.answer:                      # Estimated Gestational Age: a (weeks, days) pair
        got = _weeks_days(boxed)
        if got is None:
            return 0.0, {"valid": False, "parsed": boxed, "reason": "not a (weeks, days) pair"}
        ok = got == _weeks_days(ex.answer)
        return float(ok), {"valid": True, "parsed": got, "reason": "in range" if ok else "out of range"}
    if ex.output_type == "date":
        got = _date(boxed)
        if got is None:
            return 0.0, {"valid": False, "parsed": boxed, "reason": "not an M/D/Y date"}
        ok = _date(ex.lower) <= got <= _date(ex.upper)
        return float(ok), {"valid": True, "parsed": got.strftime("%m/%d/%Y"), "reason": "in range" if ok else "out of range"}
    nums = _NUM.findall(boxed.replace(",", ""))
    if len(nums) != 1:
        return 0.0, {"valid": False, "parsed": boxed, "reason": f"{len(nums)} numbers in the box"}
    v = float(nums[0])
    ok = float(ex.lower) <= v <= float(ex.upper)
    return float(ok), {"valid": True, "parsed": v, "reason": "in range" if ok else "out of range"}
