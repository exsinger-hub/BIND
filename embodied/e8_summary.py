"""E8 clarification trade-off on the memory-grounded split (REF / ATT / AMB, same utterance).
Per unit: a correct action on a REF/ATT unit is +1, a wrong action (including ANY action on an AMB unit, whose
reading is undetermined) costs lambda, asking is 0. U_lambda = (correct - lambda * wrong) / units, mean over VLMs.
Policies come from intent files (pred in fixed/dynamic/ask); --sc-file adds self-consistency thresholds from votes.
usage: python e8_summary.py --policy name=file ... [--sc name=file] --files <raw> --instr instructions_grounded.jsonl --out e8.csv
"""
import argparse
import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path

import diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", nargs="+", default=[])
    ap.add_argument("--sc", nargs="*", default=[], help="name=file with votes; asks when agreement < k/n")
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--instr", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("rv", str(Path(diag.__file__).with_name("run_vlm.py")))
    rv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rv)
    instr = {json.loads(l)["episode_id"]: json.loads(l) for l in open(a.instr, encoding="utf-8")}
    G0 = diag.load(a.files, instr, None, rv)
    policies = {}
    for p in a.policy:
        name, fn = p.split("=", 1)
        policies[name] = {(m["episode_id"], m["intent"]): m["pred"] for m in map(json.loads, open(fn))}
    for p in a.sc:
        name, fn = p.split("=", 1)
        M = [json.loads(l) for l in open(fn)]
        for k in (5, 4):
            pol = {}
            for m in M:
                v = Counter(m["votes"]).most_common(1)[0]
                pol[(m["episode_id"], m["intent"])] = v[0] if v[1] >= k else "ask"
            policies[f"{name}_agree{k}of5"] = pol
    universe = [(e, i) for e in instr for i in ("fixed", "dynamic", "ambig")]
    policies["oracle"] = {k: ("ask" if k[1] == "ambig" else k[1]) for k in universe}
    rows = []
    for name, pol in policies.items():
        # a policy without an entry for an AMB item acts on it (never-ask files only cover fixed/dynamic)
        pol = {k: pol.get(k, "fixed" if k[1] == "ambig" else "?") for k in universe}
        tot = Counter(k[1] for k in universe)
        ask = Counter(k[1] for k, v in pol.items() if v == "ask")
        per = []
        for tag, R in sorted(G0.items()):
            rel = [r for r in R if r["cond"] == "rel" and r["intent"] in ("fixed", "dynamic")]
            ok = wrong = 0
            for r in rel:
                p = pol.get((r["episode_id"], r["intent"]))
                if p == "ask":
                    continue
                d = r["out"].get("draft")
                t = diag.A.tgt(d if (p == "fixed" and d) else r["out"].get("P0_naive"))
                ok += t == r["gold"]
                wrong += t != r["gold"]
            per.append((ok, wrong, len(rel)))
        n_det = sum(n for _, _, n in per) / len(per)
        ok = sum(o for o, _, _ in per) / len(per)
        wrong_det = sum(w for _, w, _ in per) / len(per)
        wrong_amb = tot["ambig"] - ask["ambig"]  # any action on an undetermined request counts as unsupported
        N = n_det + tot["ambig"]
        row = dict(policy=name, ask_ref=ask["fixed"] / tot["fixed"], ask_att=ask["dynamic"] / tot["dynamic"],
                   ask_amb=ask["ambig"] / tot["ambig"], acc_det=ok / n_det, sel_acc=ok / max(1, ok + wrong_det),
                   ask_rate=sum(ask.values()) / len(universe), correct=ok / N, wrong=(wrong_det + wrong_amb) / N)
        for lam in (0.0, 0.5, 1.0, 2.0):
            row[f"U_{lam}"] = (ok - lam * (wrong_det + wrong_amb)) / N
        rows.append(row)
        print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
