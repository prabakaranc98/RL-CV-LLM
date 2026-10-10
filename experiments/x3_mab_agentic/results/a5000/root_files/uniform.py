import json, re
d = json.load(open("data/medagentbench/test_data_v2.json"))
for t in range(1, 11):
    xs = [x for x in d if x["id"].split("_")[0] == f"task{t}"]
    ctx = {x["context"] for x in xs}
    ins = {x["instruction"].replace(x.get("eval_MRN") or "@@", "<MRN>") for x in xs}
    print(f"task{t}: n={len(xs)} contexts={len(ctx)} instructions(after MRN strip)={len(ins)} no-MRN={sum('eval_MRN' not in x for x in xs)}")
t1 = [x for x in d if x["id"].startswith("task1_")]
for x in t1[:3] + [x for x in t1 if x["sol"] == ["Patient not found"]]:
    print("  ", x["id"], x["sol"], x.get("eval_MRN"), "|", x["instruction"][:110])
