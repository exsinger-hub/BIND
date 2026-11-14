"""Summarize intent-module ablations: intent accuracy per construction label + end-to-end Ours accuracy with the
grounding outputs held fixed (all five VLMs), via diag.load.
usage: python e2_summary.py --modes modes_e2/*.jsonl --files <grounded raw files> --instr instructions_grounded.jsonl --out e2.csv
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
    ap.add_argument("--modes", nargs="+", required=True)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--instr", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("rv", str(Path(diag.__file__).with_name("run_vlm.py")))
    rv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rv)
    instr = {json.loads(l)["episode_id"]: json.loads(l) for l in open(a.instr, encoding="utf-8")}
    rows = []
    for mf in a.modes:
        M = [json.loads(l) for l in open(mf)]
        c = Counter((m["intent"], m["pred"]) for m in M)
        tot = Counter(m["intent"] for m in M)
        ext = {(m["episode_id"], m["intent"]): m["pred"] for m in M}
        G = diag.load(a.files, instr, ext, rv)
        row = dict(variant=Path(mf).stem, ref_acc=c[("fixed", "fixed")] / max(1, tot["fixed"]),
                   att_acc=c[("dynamic", "dynamic")] / max(1, tot["dynamic"]),
                   intent_acc=(c[("fixed", "fixed")] + c[("dynamic", "dynamic")]) / max(1, tot["fixed"] + tot["dynamic"]),
                   ask_ref=c[("fixed", "ask")] / max(1, tot["fixed"]), ask_att=c[("dynamic", "ask")] / max(1, tot["dynamic"]),
                   ask_amb=c[("ambig", "ask")] / max(1, tot["ambig"]),
                   ptok=sum(m.get("ptok", 0) for m in M) / len(M), ntok=sum(m.get("ntok", 0) for m in M) / len(M))
        accs = []
        for tag, R in sorted(G.items()):
            rel = [r for r in R if r["cond"] == "rel" and r["intent"] in ("fixed", "dynamic")]
            acc = sum(diag.A.tgt(r["out"].get("Ours")) == r["gold"] for r in rel) / len(rel)
            row[f"e2e_{tag}"] = acc
            accs.append(acc)
        row["e2e_mean"] = sum(accs) / len(accs)
        rows.append(row)
        print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
