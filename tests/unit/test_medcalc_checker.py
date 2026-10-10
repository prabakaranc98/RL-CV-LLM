"""The MedCalc reward checker: it must accept every ground truth and reject the obvious hacks.

Run on the pod:  pytest -q tests/unit/test_medcalc_checker.py
"""
import collections

from jaxpt.envs import medcalc


def test_ground_truth_always_scores_one():
    """Writing the gold answer in the expected format must give reward 1.0, for every row of both splits."""
    failures = collections.Counter()
    n = 0
    for source, split in (("verified", "train"), ("verified", "test"), ("verified", "one_shot"), ("v1.2", "train"), ("v1.2", "test")):
        for ex in medcalc.load(split, source):
            r, info = medcalc.reward(f"... The answer is \\boxed{{{ex.answer}}}.", ex)
            n += 1
            if r != 1.0:
                failures[(source, ex.id, ex.calculator, ex.answer, info["reason"])] += 1
    print(f"\n  ground truth accepted for {n - sum(failures.values())}/{n} rows")
    for k, v in failures.most_common(10):
        print("   rejected:", k, v)
    assert not failures


def test_rejects_hacks_and_bad_formats():
    ex = next(e for e in medcalc.load("test") if e.output_type == "decimal")
    lo, hi = float(ex.lower), float(ex.upper)
    mid = (lo + hi) / 2
    cases = {
        "no box": (f"The answer is {mid}.", 0.0),
        "range in the box": (f"\\boxed{{{lo - 1}-{hi + 1}}}", 0.0),
        "two different boxes": (f"\\boxed{{{lo - 50}}} or \\boxed{{{mid}}}", 0.0),
        "same box twice": (f"\\boxed{{{mid}}} ... so \\boxed{{{mid}}}", 1.0),
        "units in the box": (f"\\boxed{{{mid} mL/min}}", 1.0),
        "thousands separator": (f"\\boxed{{{mid:,.3f}}}", 1.0),
        "just outside the range": (f"\\boxed{{{hi * 1.01 + 0.01}}}", 0.0),
    }
    for name, (text, want) in cases.items():
        got, info = medcalc.reward(text, ex)
        print(f"  {name:24s} reward {got}  ({info['reason']})")
        assert got == want, name


def test_values_text_renders_every_row():
    """The gold-values input must render for every row (no parse errors, never empty)."""
    bad = []
    for split in ("train", "test", "one_shot"):
        for ex in medcalc.load(split):
            try:
                assert medcalc.values_text(ex).strip()
            except Exception as err:
                bad.append((ex.id, ex.calculator, repr(err)[:80]))
    ex = next(e for e in medcalc.load("test") if e.calculator.startswith("Creatinine Clearance"))
    print("\n  example values block:\n" + "\n".join("    " + l for l in medcalc.problem_block(ex, "values").splitlines()))
    assert not bad, bad[:5]
