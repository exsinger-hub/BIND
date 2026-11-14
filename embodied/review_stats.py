"""Statistics requested by the internal review (memory-grounded split, grounding outputs of all five VLMs).
(1) paired scene-cluster bootstrap CIs of accuracy differences (pooled over VLMs, unit = VLM x episode x intent);
(2) independent-axis counts: read-time draft correct?, route correct?, final correct? per routed method;
(3) raw drift / wrong-freeze rates of Ours (no precedence);
(4) expected accuracy when only a fraction p of retries follow a referent-changing write: p*acc_rel + (1-p)*acc_unr.
usage: python review_stats.py --files <raw> --instr instructions_grounded.jsonl --llm modes.jsonl
"""
import argparse
import importlib.util
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import diag


def boot_diff(units, fa, fb, reps=2000, seed=2026):
    g = defaultdict(list)
    for u in units:
        g[u["episode_id"].split("_s")[0]].append(u)
    ks, rng, vals = list(g), random.Random(seed), []
    for _ in range(reps):
        s = [u for k in (rng.choice(ks) for _ in ks) for u in g[k]]
        vals.append((sum(map(fa, s)) - sum(map(fb, s))) / len(s))
    vals.sort()
    d = (sum(map(fa, units)) - sum(map(fb, units))) / len(units)
    return 100 * d, 100 * vals[int(.025 * reps)], 100 * vals[int(.975 * reps) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--instr", required=True)
    ap.add_argument("--llm", required=True)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("rv", str(Path(diag.__file__).with_name("run_vlm.py")))
    rv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rv)
    instr = {json.loads(l)["episode_id"]: json.loads(l) for l in open(a.instr, encoding="utf-8")}
    llm = {(m["episode_id"], m["intent"]): m["pred"] for m in map(json.loads, open(a.llm))}
    rel, unr = [], []
    for f in a.files:
        G = diag.load([f], instr, llm, rv)
        idr = {}
        for tag, R in G.items():
            for r in R:
                d = diag.A.tgt(r["out"].get("draft"))
                notes = instr[r["episode_id"]][r["intent"]]["notes"]
                p = "fixed" if d and any(re.search(rf"\b{d}\b", n) for n in notes) else "dynamic"
                r["out"]["BIND_ID"] = r["out"].get("draft") if p == "fixed" and r["out"].get("draft") else r["out"].get("P0_naive")
                r["route"]["BIND_ID"] = p
                r["vlm"] = tag
                (rel if r["cond"] == "rel" else unr).append(r) if r["intent"] in ("fixed", "dynamic") else None
    ok = lambda m: (lambda r: diag.A.tgt(r["out"].get(m)) == r["gold"])
    print("== paired differences (pp) with 95% scene-cluster bootstrap CI, pooled over 5 VLMs, n =", len(rel))
    for a_, b_ in [("Ours", "BIND_OWN"), ("Ours", "P0_naive"), ("Ours", "E"), ("Ours", "FREEZE"), ("Ours", "P0"),
                   ("BIND_ID", "Ours"), ("Oracle", "Ours"), ("BIND_ID", "Oracle"), ("Ours_2V", "Ours")]:
        d, lo, hi = boot_diff(rel, ok(a_), ok(b_))
        print(f"  {a_:8s} - {b_:8s}: {d:+5.1f} [{lo:+5.1f}, {hi:+5.1f}]")
    print("== independent axes (pooled): counts of (draft ok, route ok, final ok)")
    for m in ("Ours", "BIND_ID", "BIND_OWN"):
        c = Counter((diag.A.tgt(r["out"].get("draft")) == r["ref_B"], r["route"][m] == r["intent"],
                     diag.A.tgt(r["out"].get(m)) == r["gold"]) for r in rel)
        tot = len(rel)
        print(f"  {m}: draft-ok {100*sum(v for k,v in c.items() if k[0])/tot:.1f}%  route-ok {100*sum(v for k,v in c.items() if k[1])/tot:.1f}%  final-ok {100*sum(v for k,v in c.items() if k[2])/tot:.1f}%")
        for k in sorted(c):
            print("     draft_ok=%d route_ok=%d final_ok=%d : %d" % (k[0], k[1], k[2], c[k]))
    fx = [r for r in rel if r["intent"] == "fixed"]
    dy = [r for r in rel if r["intent"] == "dynamic"]
    for m in ("Ours", "BIND_ID"):
        dr = sum(diag.A.tgt(r["out"].get(m)) == r["ref_C"] for r in fx) / len(fx)
        fr = sum(diag.A.tgt(r["out"].get(m)) == r["ref_B"] for r in dy) / len(dy)
        print(f"== raw rates {m}: drift (REF units ending on C-referent) {100*dr:.2f}%  wrong freeze (ATT ending on B-referent) {100*fr:.2f}%")
    print("== expected accuracy vs prevalence p of referent-changing writes (acc_rel, acc_unr)")
    for m in ("P0_naive", "P0", "E", "FREEZE", "BIND_OWN", "Ours", "BIND_ID"):
        ar = sum(map(ok(m), rel)) / len(rel)
        au = sum(map(ok(m), unr)) / len(unr)
        print(f"  {m:9s} rel {100*ar:.1f} unr {100*au:.1f}  p=0.1: {100*(0.1*ar+0.9*au):.1f}  p=0.3: {100*(0.3*ar+0.7*au):.1f}  p=0.5: {100*(0.5*ar+0.5*au):.1f}")


if __name__ == "__main__":
    main()
