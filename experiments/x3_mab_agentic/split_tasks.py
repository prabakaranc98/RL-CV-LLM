"""Patient-disjoint split of the official MedAgentBench v2 tasks: train on two thirds of the patients, report on the rest.

All clinical data on the FHIR server belongs to the 98 test patients (the other 597 have demographics only; see
make_tasks.py), so lab tasks can only be trained on test patients. A third of them is held out with all their tasks.
The seed is the first of 2,000 whose held-out set has the most even count per task type, chosen on type balance only,
never on results. The two task1 "Patient not found" cases have no patient and go to train.

Writes <out>/train_official.json, <out>/heldout_official.json (official task format) and <out>/split.json.
    python split_tasks.py --mab /root/MedAgentBench --out /root/mab_train
"""
import argparse
import collections
import json
import os
import random


def split(tasks, seed):
    patients = sorted({x["eval_MRN"] for x in tasks if x.get("eval_MRN")})
    random.Random(seed).shuffle(patients)
    held = set(patients[: len(patients) // 3])
    return held, [x for x in tasks if x.get("eval_MRN") not in held], [x for x in tasks if x.get("eval_MRN") in held]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mab", default="/root/MedAgentBench")
    ap.add_argument("--out", default="/root/mab_train")
    args = ap.parse_args()
    tasks = json.load(open(f"{args.mab}/data/medagentbench/test_data_v2.json"))
    best = None
    for seed in range(2000):
        held, train, test = split(tasks, seed)
        per = collections.Counter(x["id"].split("_")[0] for x in test)
        spread = max(per.values()) - min(per.values()) if len(per) == 10 else 1 << 30
        if best is None or spread < best[0]:
            best = (spread, seed)
    seed = best[1]
    held, train, test = split(tasks, seed)
    os.makedirs(args.out, exist_ok=True)
    json.dump(train, open(f"{args.out}/train_official.json", "w"), indent=1)
    json.dump(test, open(f"{args.out}/heldout_official.json", "w"), indent=1)
    per = lambda xs: dict(sorted(collections.Counter(x["id"].split("_")[0] for x in xs).items(), key=lambda kv: int(kv[0][4:])))
    json.dump({"seed": seed, "heldout_patients": sorted(held), "train_tasks": len(train), "heldout_tasks": len(test),
               "heldout_per_type": per(test)}, open(f"{args.out}/split.json", "w"), indent=1)
    print(f"seed {seed}: held-out patients {len(held)}, train tasks {len(train)}, held-out tasks {len(test)}")
    print("held-out per type", per(test))


if __name__ == "__main__":
    main()
