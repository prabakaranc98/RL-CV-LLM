import json
d = json.load(open("data/medagentbench/test_data_v2.json"))
print("fields:", sorted(d[0].keys()))
seen = set()
for x in d:
    t = x["id"].split("_")[0]
    if t in seen:
        continue
    seen.add(t)
    print()
    print(f"[{t}] {x['instruction']}")
    print(f"   context: {x.get('context', '')[:220]}")
    print(f"   sol: {x.get('sol')}   eval_MRN: {x.get('eval_MRN')}")
