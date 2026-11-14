"""Offline routing analysis: combine configs by (oracle / predicted) binding mode."""
import json, sys
from collections import defaultdict
R = [json.loads(l) for l in open(sys.argv[1])]
t = lambda op: (op or {}).get("target")
def acc(pred_mode, ref_cfg="FREEZE", att_cfg="P0_naive", units=None):
    U = [r for r in R if r["cond"] == "rel" and r["intent"] in ("fixed", "dynamic")] if units is None else units
    ok = 0
    for r in U:
        m = pred_mode(r)
        out = r["out"].get(ref_cfg) if m == "fixed" else r["out"].get(att_cfg)
        ok += t(out) == r["gold"]
    return ok / len(U)
oracle = lambda r: "fixed" if r["intent"] == "fixed" else "dynamic"
selftag = lambda r: "fixed" if r["kind"] == "fixed" else "dynamic"
erw = lambda r: "fixed" if r.get("rewritten") and "obj" in r["rewritten"].lower() else "dynamic"
U = [r for r in R if r["cond"] == "rel" and r["intent"] in ("fixed", "dynamic")]
print("n rel units", len(U))
for name, f in [("oracle", oracle), ("self-tag", selftag), ("E-rewrite has id", erw)]:
    ma = sum(f(r) == oracle(r) for r in U) / len(U)
    print(f"{name:18s} mode-acc {ma:.3f} | route FREEZE/P0_naive {acc(f):.3f} | FREEZE/E {acc(f, 'FREEZE', 'E'):.3f}")
for c in ["P0_naive", "P0", "E", "P1", "FREEZE"]:
    print(f"single cfg {c:9s} acc {sum(t(r['out'].get(c)) == r['gold'] for r in U)/len(U):.3f}")
# ceiling limited by initial binding: fixed units where draft correct
fx = [r for r in U if r["intent"] == "fixed"]
print("draft correct on fixed:", sum(t(r["out"]["draft"]) == r["ref_B"] for r in fx) / len(fx))
dy = [r for r in U if r["intent"] == "dynamic"]
print("P0_naive correct on dynamic:", sum(t(r["out"]["P0_naive"]) == r["gold"] for r in dy) / len(dy))
