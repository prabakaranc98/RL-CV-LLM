import json, sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    if "eval_step" in r:
        print(f"EVAL step {r['eval_step']:2d}: held-out official {100*r['heldout_official']:.1f}%  gated {100*r['heldout_gated']:.1f}%  {r['statuses']}")
    else:
        n = sum(r["statuses"].values())
        print(f"step {r['step']:2d}: reward {r['reward_mean']:.3f}  right-lookup {100*r['lookup_rate']:5.1f}%  finished {100*r['statuses'].get('completed',0)/n:5.1f}%  "
              f"official {100*r['official_rate']:4.1f}%  gated {100*r['gated_rate']:4.1f}%  signal {r['signal_groups']}/8  ent {r['entropy']:.3f}")
