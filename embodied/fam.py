"""E6: accuracy by description family and room for each method (memory-grounded split), long CSV.
usage: python fam.py --files ... --instr ... --intent ... --out fam.csv
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path

import diag

METHODS = {"P0_naive": "Re-plan (current view)", "P0": "Retry (B+C+log)", "E": "Explicitize", "FREEZE": "Freeze ID",
           "BIND_OWN": "BIND (own intent)", "Ours": "Ours", "Oracle": "Oracle intent"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--instr")
    ap.add_argument("--intent")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("rv", str(Path(diag.__file__).with_name("run_vlm.py")))
    rv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rv)
    instr = {json.loads(l)["episode_id"]: json.loads(l) for l in open(a.instr, encoding="utf-8")} if a.instr else {}
    ext = {(m["episode_id"], m["intent"]): m["pred"] for m in map(json.loads, open(a.intent))} if a.intent else None
    rows = []
    for tag, R in sorted(diag.load(a.files, instr, ext, rv).items()):
        rel = [r for r in R if r["cond"] == "rel" and r["intent"] in ("fixed", "dynamic")]
        for by in ("family", "room"):
            keys = sorted({("left/right" if r[by] in ("left", "right") else r[by]) for r in rel})
            for k in keys:
                sub = [r for r in rel if ("left/right" if r[by] in ("left", "right") else r[by]) == k]
                for m, name in METHODS.items():
                    ok = [diag.A.tgt(r["out"].get(m)) == r["gold"] for r in sub]
                    lo, hi = diag.boot_ci(sub, lambda r, m=m: diag.A.tgt(r["out"].get(m)) == r["gold"])
                    rows.append(dict(model=tag, by=by, group=k, method=name, acc=sum(ok) / len(ok), ci_low=lo,
                                     ci_high=hi, n=len(sub)))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        if r["method"] in ("Ours", "Re-plan (current view)", "Freeze ID") and r["by"] == "family":
            print(f'{r["model"]:14s} {r["group"]:10s} {r["method"]:24s} {100*r["acc"]:5.1f} n={r["n"]}')


if __name__ == "__main__":
    main()
