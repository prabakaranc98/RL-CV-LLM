"""Training tasks for agentic RL on MedAgentBench, built from the benchmark's own templates on patients the test never uses.

MedAgentBench ships test files only (v1: 100 tasks, v2: 300). Each task type is one fixed instruction + context with the
patient's MRN substituted (task1: a name + DOB instead), and the official grader (refsol.py) reads only the MRN, the task
type and, for task1, the expected MRN; every other reference answer it computes live from the FHIR server. So the same
templates on unseen patients give training tasks graded by the official logic, with no test patient in them.

Checks (printed, and written to <out>/audit.json), run on the generated tasks and on the official 300:
  * oracle: an agent that does exactly what the task asks (the lookup GET, the required POSTs, the reference answer),
    computed here independently of refsol.py, must be graded correct by refsol.py on every task;
  * shortcut: share of tasks where FINISH([]) with no action is graded correct, per task type;
  * label mix: per type, how often the measurement is missing / normal / needs an order.

Run on the MedAgentBench pod (its venv has the grader's dependencies):
    python make_tasks.py --mab /root/MedAgentBench --out /root/mab_train
"""
import argparse
import collections
import importlib
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

STAMP = "2023-11-13T10:15:00+00:00"                     # the benchmark's "now"
NOW = datetime.fromisoformat(STAMP)
ACCEPTED = "POST request accepted and executed successfully."
NAME_DOB = re.compile(r"name (.+?) and DOB of (\d{4}-\d{2}-\d{2})")


class Turn:
    def __init__(self, role, content):
        self.role, self.content = role, content


class Result:
    """What refsol.py reads from an AgentBench TaskOutput: the FINISH payload and the chat history."""
    def __init__(self, result, history=()):
        self.result, self.history = result, list(history)


def cached(get):
    """The FHIR server is read-only during evaluation (POSTs are never sent), so a URL always returns the same data."""
    cache, lock = {}, threading.Lock()

    def send(url, params=None, headers=None):
        with lock:
            if url in cache:
                return cache[url]
        for _ in range(3):
            r = get(url, params, headers)
            if "data" in r:
                with lock:
                    cache[url] = r
                return r
        return r
    return send


def all_patients(get, base):
    out, url = [], f"{base}Patient?_count=200&_format=json"
    while url:
        page = json.loads(get(url)["data"])
        for e in page.get("entry", []):
            p = e["resource"]
            name = (p.get("name") or [{}])[0]
            out.append({"mrn": p["identifier"][0]["value"], "given": " ".join(name.get("given", [])),
                        "family": name.get("family", ""), "dob": p.get("birthDate")})
        nxt = next((l["url"] for l in page.get("link", []) if l["relation"] == "next"), None)
        url = f"{base.rstrip('/')}?{nxt.split('?', 1)[1]}" if nxt else None      # paging links name the internal host
        if url and "_format=json" not in url:
            url += "&_format=json"
    return out


def observations(get, base, mrn, code):
    url = f"{base}Observation?patient={mrn}&code={code}&_count=5000&_format=json"   # the grader's own URL
    entries = json.loads(get(url)["data"]).get("entry", [])
    return url, [(datetime.fromisoformat(e["resource"]["effectiveDateTime"]), e["resource"]["valueQuantity"]["value"],
                  e["resource"]["effectiveDateTime"]) for e in entries]


def latest(obs, since=None):
    obs = [o for o in obs if since is None or o[0] >= since]
    return max(obs, key=lambda o: o[0]) if obs else None


def age(dob):
    d = datetime.strptime(dob, "%Y-%m-%d")
    return 2023 - d.year - ((11, 13) < (d.month, d.day))


def subject(mrn):
    return {"reference": f"Patient/{mrn}"}


def plan(case, base, get, patients_by_mrn):
    """Independent reference: (GET urls the task needs, POSTs it needs, FINISH answer, label for the audit)."""
    t, mrn = case["id"].split("_")[0], case.get("eval_MRN")
    if t == "task1":
        name, dob = NAME_DOB.search(case["instruction"]).groups()
        given, family = name.rsplit(" ", 1)
        return [f"{base}Patient?given={given}&family={family}&birthdate={dob}"], [], case["sol"], \
            "not-found" if case["sol"] == ["Patient not found"] else "found"
    if t == "task2":
        return [f"{base}Patient?identifier={mrn}"], [], [age(patients_by_mrn[mrn]["dob"])], "n/a"
    if t == "task3":
        return [], [(f"{base}Observation", {
            "resourceType": "Observation", "status": "final", "effectiveDateTime": STAMP, "valueString": "118/77 mmHg",
            "category": [{"coding": [{"system": "http://hl7.org/fhir/observation-category", "code": "vital-signs",
                                      "display": "Vital Signs"}]}],
            "code": {"text": "BP"}, "subject": subject(mrn)})], [], "n/a"
    if t == "task8":
        note = ("Situation: acute left knee injury, Background: radiology report indicates ACL tear. Assessment: ACL tear "
                "grade II. Recommendation: request for Orthopedic service to evaluate and provide management "
                "recommendations.")
        return [], [(f"{base}ServiceRequest", {
            "resourceType": "ServiceRequest", "status": "active", "intent": "order", "priority": "stat",
            "code": {"coding": [{"system": "http://snomed.info/sct", "code": "306181000000106"}]},
            "authoredOn": STAMP, "note": {"text": note}, "subject": subject(mrn)})], [], "n/a"

    code = {"task4": "MG", "task5": "MG", "task6": "GLU", "task7": "GLU", "task9": "K", "task10": "A1C"}[t]
    url, obs = observations(get, base, mrn, code)
    day = NOW - timedelta(hours=24)
    if t == "task4":
        last = latest(obs, day)
        return [url], [], [last[1] if last else -1], "value" if last else "missing"
    if t == "task6":
        recent = [o[1] for o in obs if o[0] >= day]
        return [url], [], [sum(recent) / len(recent) if recent else -1], "value" if recent else "missing"
    if t == "task7":
        last = latest(obs)
        return [url], [], [last[1] if last else -1], "value" if last else "missing"
    if t == "task5":
        last = latest(obs, day)
        if last is None or last[1] > 1.9:
            return [url], [], [last[1] if last else -1], "normal" if last else "missing"
        dose = 4 if last[1] < 1 else 2 if last[1] < 1.5 else 1
        return [url], [(f"{base}MedicationRequest", {
            "resourceType": "MedicationRequest", "status": "active", "intent": "order", "authoredOn": STAMP,
            "medicationCodeableConcept": {"coding": [{"system": "http://hl7.org/fhir/sid/ndc", "code": "0338-1715-40"}]},
            "dosageInstruction": [{"route": "IV", "doseAndRate": [{"doseQuantity": {"value": dose, "unit": "g"},
                                                                   "rateQuantity": {"value": dose, "unit": "h"}}]}],
            "subject": subject(mrn)})], [last[1]], "order"
    if t == "task9":
        last = latest(obs)
        if last is None or last[1] >= 3.5:
            return [url], [], [last[1] if last else -1], "normal" if last else "missing"
        return [url], [
            (f"{base}MedicationRequest", {
                "resourceType": "MedicationRequest", "status": "active", "intent": "order", "authoredOn": STAMP,
                "medicationCodeableConcept": {"coding": [{"system": "http://hl7.org/fhir/sid/ndc", "code": "40032-917-01"}]},
                "dosageInstruction": [{"route": "oral", "doseAndRate": [{"doseQuantity": {
                    "value": round((3.5 - last[1]) / 0.1 * 10, 4), "unit": "mEq"}}]}],
                "subject": subject(mrn)}),
            (f"{base}ServiceRequest", {
                "resourceType": "ServiceRequest", "status": "active", "intent": "order", "priority": "stat",
                "code": {"coding": [{"system": "http://loinc.org", "code": "2823-3"}]}, "authoredOn": STAMP,
                "occurrenceDateTime": "2023-11-14T08:00:00+00:00", "subject": subject(mrn)})], [last[1]], "order"
    if t == "task10":
        last = latest(obs)
        answer = [last[1], last[2]] if last else [-1]
        if last and last[0] >= datetime.fromisoformat("2022-11-13T10:15:00+00:00"):
            return [url], [], answer, "recent"
        return [url], [(f"{base}ServiceRequest", {
            "resourceType": "ServiceRequest", "status": "active", "intent": "order", "priority": "stat",
            "code": {"coding": [{"system": "http://loinc.org", "code": "4548-4"}]}, "authoredOn": STAMP,
            "subject": subject(mrn)})], answer, "old" if last else "missing"
    raise ValueError(t)


def oracle_result(gets, posts, answer):
    history = []
    for url in gets:
        history += [Turn("agent", f"GET {url}"), Turn("user", "Here is the response from the GET request:\n{...}")]
    for url, payload in posts:
        history += [Turn("agent", f"POST {url}\n{json.dumps(payload)}"), Turn("user", ACCEPTED)]
    history.append(Turn("agent", f"FINISH({json.dumps(answer)})"))
    return Result(json.dumps(answer), history)


def make_training_tasks(test, patients, rng, not_found_share):
    templates, contexts = {}, {}
    for x in test:
        t = x["id"].split("_")[0]
        contexts.setdefault(t, x["context"])
        if t == "task1":
            templates.setdefault("task1-" + ("missing" if x["sol"] == ["Patient not found"] else "found"), x["instruction"])
        else:
            templates.setdefault(t, x["instruction"].replace(x["eval_MRN"], "{MRN}"))
    known = {(p["given"], p["family"], p["dob"]) for p in patients}
    tasks = []
    for k, p in enumerate(patients):
        if rng.random() < not_found_share:                     # a name + DOB that matches no patient on the server
            while True:
                q, r = rng.sample(patients, 2)
                if (q["given"], r["family"], q["dob"]) not in known:
                    break
            ins = NAME_DOB.sub(f"name {q['given']} {r['family']} and DOB of {q['dob']}", templates["task1-missing"])
            tasks.append({"id": f"task1_train{k}", "instruction": ins, "context": contexts["task1"],
                          "sol": ["Patient not found"]})
        else:
            ins = NAME_DOB.sub(f"name {p['given']} {p['family']} and DOB of {p['dob']}", templates["task1-found"])
            tasks.append({"id": f"task1_train{k}", "instruction": ins, "context": contexts["task1"],
                          "sol": [p["mrn"]], "eval_MRN": p["mrn"]})
        for t in range(2, 11):
            tasks.append({"id": f"task{t}_train{k}", "instruction": templates[f"task{t}"].replace("{MRN}", p["mrn"]),
                          "context": contexts[f"task{t}"], "sol": None, "eval_MRN": p["mrn"]})
    return tasks


def audit(tasks, refsol, base, get, patients_by_mrn, workers):
    def one(case):
        t = case["id"].split("_")[0]
        gets, posts, answer, label = plan(case, base, get, patients_by_mrn)
        grade = getattr(refsol, t)
        return t, label, grade(case, oracle_result(gets, posts, answer), base) is True, \
            grade(case, Result("[]"), base) is True
    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(one, tasks))
    by = collections.defaultdict(lambda: {"n": 0, "oracle": 0, "shortcut": 0, "labels": collections.Counter()})
    for t, label, ok, short in rows:
        b = by[t]
        b["n"] += 1; b["oracle"] += ok; b["shortcut"] += short; b["labels"][label] += 1
    return {t: {**b, "labels": dict(b["labels"])} for t, b in sorted(by.items(), key=lambda kv: int(kv[0][4:]))}


def show(name, a):
    n = sum(b["n"] for b in a.values())
    print(f"\n{name}: {n} tasks | oracle graded correct {sum(b['oracle'] for b in a.values())}/{n} | "
          f"FINISH([]) graded correct {sum(b['shortcut'] for b in a.values())}/{n}")
    for t, b in a.items():
        print(f"  {t:7} n={b['n']:4}  oracle {b['oracle']:4}  shortcut {b['shortcut']:4} ({100 * b['shortcut'] / b['n']:5.1f}%)"
              f"  {b['labels']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mab", default="/root/MedAgentBench")
    ap.add_argument("--out", default="/root/mab_train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    sys.path.insert(0, args.mab)
    os.chdir(args.mab)
    refsol = importlib.import_module("src.server.tasks.medagentbench.refsol")
    base = re.search(r'fhir_api_base:\s*"([^"]+)"', open("configs/tasks/medagentbench.yaml").read()).group(1)
    get = cached(refsol.send_get_request)
    refsol.send_get_request = get                                   # the grader shares the cache
    refsol.print = lambda *a, **k: None                            # silence the grader's per-task prints

    test = json.load(open("data/medagentbench/test_data_v2.json"))
    test_v1 = json.load(open("data/medagentbench/test_data_v1.json"))
    test_mrns = {x["eval_MRN"] for x in test + test_v1 if x.get("eval_MRN")}
    patients = all_patients(get, base)
    by_mrn = {p["mrn"]: p for p in patients}
    assert test_mrns <= set(by_mrn), f"test MRNs missing from the server: {sorted(test_mrns - set(by_mrn))[:5]}"
    t1 = [x for x in test if x["id"].startswith("task1_") and x.get("eval_MRN")]
    mism = [x["id"] for x in t1 if NAME_DOB.search(x["instruction"]).groups()
            != (f"{by_mrn[x['eval_MRN']]['given']} {by_mrn[x['eval_MRN']]['family']}", by_mrn[x["eval_MRN"]]["dob"])]
    print(f"server patients {len(patients)} | test patients (v1 + v2) {len(test_mrns)} | "
          f"task1 test names that don't match given + family + DOB: {mism}")

    train_patients = [p for p in patients if p["mrn"] not in test_mrns]
    rng = random.Random(args.seed)
    n_missing = sum(x["sol"] == ["Patient not found"] for x in test if x["id"].startswith("task1_"))
    tasks = make_training_tasks(test, train_patients, rng, n_missing / 30)
    rng.shuffle(tasks)

    a_test = audit(test, refsol, base, get, by_mrn, args.workers)
    show("official test (v2)", a_test)
    a_train = audit(tasks, refsol, base, get, by_mrn, args.workers)
    show(f"training ({len(train_patients)} patients not in v1 or v2)", a_train)

    os.makedirs(args.out, exist_ok=True)
    json.dump(tasks, open(f"{args.out}/train_tasks.json", "w"), indent=1)
    json.dump({"train_patients": [p["mrn"] for p in train_patients], "test_patients": sorted(test_mrns)},
              open(f"{args.out}/patient_split.json", "w"), indent=1)
    json.dump({"test": a_test, "train": a_train}, open(f"{args.out}/audit.json", "w"), indent=1)
    print(f"\nwrote {len(tasks)} tasks to {args.out}/train_tasks.json")


if __name__ == "__main__":
    main()
