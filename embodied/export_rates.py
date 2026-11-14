"""Export pre-aggregated rates (+ scene-cluster bootstrap CI) as long CSV for plotting.

usage: python export_rates.py --files res/both/x.jsonl ... --instr instr.jsonl --label <setting> --out rates.csv [--append]
Rows: setting, model, method, task (metric), success (rate), ci_low, ci_high, n
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import analyze_vlm as A  # noqa: E402

METHODS = {"P0_naive": "Re-plan (current view)", "P0": "Retry (B+C+log)", "P0_aware": "Retry + intent prompt",
           "H": "Event history", "E": "Explicitize", "P1": "Self-tag binding", "FREEZE": "Freeze ID",
           "KW": "Keyword routing", "BIND": "Ours"}
METRICS = {"drift": "Drift", "freeze": "Wrong freeze", "acc_rel": "Accuracy", "named": "Named-control drift"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--instr")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--intent", help="jsonl of external intent predictions (episode_id,intent,pred) -> Ours")
    ap.add_argument("--own-bind-label", default="BIND (own intent)")
    a = ap.parse_args()
    sys.argv = ["analyze_vlm.py", *a.files] + (["--instr", a.instr] if a.instr else [])
    rows = []
    # reuse analyze_vlm's loading/re-parsing by calling its main-time logic through a small shim
    import importlib.util
    spec = importlib.util.spec_from_file_location("rv", str(Path(__file__).with_name("run_vlm.py")))
    rv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rv)
    instr = {}
    if a.instr:
        for l in open(a.instr, encoding="utf-8"):
            r = json.loads(l)
            instr[r["episode_id"]] = r
    ext = None
    if a.intent:
        ext = {}
        for l in open(a.intent):
            m = json.loads(l)
            ext[(m["episode_id"], m["intent"])] = m["pred"]
        METHODS["BIND_OWN"] = a.own_bind_label
        A.CFGS.append("BIND_OWN")
    groups = {}
    for fn in a.files:
        tag = Path(fn).name.split(".shard")[0].replace(".det", "")
        for l in open(fn):
            r = json.loads(l)
            o = r["out"]
            for cfg, txt in (r.get("raw") or {}).items():
                o[cfg] = rv.get_op(rv.parse_json(txt))
            if r.get("draft_raw") is not None:
                dobj = rv.parse_json(r["draft_raw"]) or {}
                o["draft"] = rv.get_op(dobj)
                r["kind"] = str(dobj.get("kind", "")).lower().strip()
            d = o.get("draft")
            o["FREEZE"] = d
            o["P1"] = d if (r.get("kind") == "fixed" and d) else o.get("P0")
            if r.get("mode_raw") is not None:
                mode = rv.parse_mode(r["mode_raw"])
                o["BIND"] = d if (mode == "referential" and d) else o.get("P0_naive")
            if ext is not None:
                p = ext.get((r["episode_id"], r["intent"]))
                o["BIND_OWN"] = o.get("BIND")
                o["BIND"] = d if (p == "fixed" and d) else o.get("P0_naive")
            if instr and r["episode_id"] in instr and r["intent"] in instr[r["episode_id"]]:
                txt = instr[r["episode_id"]][r["intent"]]["instruction"]
                o["KW"] = o.get("P0_naive") if A.MARK.search(txt) else d
            groups.setdefault(tag, []).append(r)
    for tag, R in groups.items():
        S = A.summarize(R)
        for cfg, mname in METHODS.items():
            for met, tname in METRICS.items():
                z = S.get(f"{met}/{cfg}")
                if not z or z["p"] is None:
                    continue
                rows.append(dict(setting=a.label, model=tag, method=mname, task=tname, success=round(z["p"], 4),
                                 ci_low=round(z["lo"], 4), ci_high=round(z["hi"], 4), n=z["n"]))
    mode = "a" if a.append and Path(a.out).exists() else "w"
    with open(a.out, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if mode == "w":
            w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows for {list(groups)} -> {a.out}")


if __name__ == "__main__":
    main()
