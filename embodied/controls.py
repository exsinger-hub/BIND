"""Construction-aware deterministic controls requested by the v6 reviews, plus REF/ATT-split accuracy for every method.
Pooled over the grounding outputs of all given VLMs (unit = VLM x episode x intent, rel condition).
  SR        : surface rule R4 (note_cues.py) routes; REF -> replay the read-time draft, ATT -> current-view re-plan
  SR+NID    : R4 routes; REF -> commit the id that a note names for an object of the target type (draft if none)
  LLM+NID   : BIND-LLM routes; REF -> noted target-type id if any, else draft (draft verification by note id)
  Oracle+NID: gold routes with the same REF resolution (upper bound of the routing x verification grid)
usage: python controls.py --files <raw> --instr instructions_grounded.jsonl --llm modes.jsonl --episodes episodes_v1
       [--json out.json]
"""
import argparse
import importlib.util
import json
import re
from pathlib import Path

import diag
from review_stats import boot_diff

ID = re.compile(r"\bobj\d+\b")


def r4(notes, noun):
    rel_id = any(noun in n.lower() and ID.search(n) for n in notes)
    rel_noid = any(noun in n.lower() and not ID.search(n) for n in notes)
    return "fixed" if rel_id else "dynamic" if rel_noid else "fixed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--instr", required=True)
    ap.add_argument("--llm", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("rv", str(Path(diag.__file__).with_name("run_vlm.py")))
    rv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rv)
    instr = {json.loads(l)["episode_id"]: json.loads(l) for l in open(a.instr, encoding="utf-8")}
    llm = {(m["episode_id"], m["intent"]): m["pred"] for m in map(json.loads, open(a.llm))}
    eps = {}
    rel = []
    for f in a.files:
        for tag, R in diag.load([f], instr, llm, rv).items():
            for r in R:
                if r["cond"] != "rel" or r["intent"] not in ("fixed", "dynamic"):
                    continue
                eid = r["episode_id"]
                if eid not in eps:
                    eps[eid] = json.loads((Path(a.episodes) / eid / "episode.json").read_text())
                ep = eps[eid]
                noun = re.sub(r"(?<!^)(?=[A-Z])", " ", ep["target_type"]).lower()
                typed = {x["id"] for x in ep["records_B"] if x["type"] == ep["target_type"]}
                notes = instr[eid][r["intent"]]["notes"]
                nid = sorted({i for n in notes for i in ID.findall(n) if i in typed})
                o, d = r["out"], r["out"].get("draft")
                d_t = diag.A.tgt(d)
                ref_res = {"target": nid[0]} if len(nid) == 1 else d
                p_id = "fixed" if d_t and any(re.search(rf"\b{d_t}\b", n) for n in notes) else "dynamic"
                routes = {"BIND_ID": p_id, "SR": r4(notes, noun), "Ours": llm.get((eid, r["intent"])),
                          "Oracle": r["intent"]}
                o["BIND_ID"] = d if p_id == "fixed" and d else o.get("P0_naive")
                o["SR"] = d if routes["SR"] == "fixed" and d else o.get("P0_naive")
                o["SR_NID"] = ref_res if routes["SR"] == "fixed" else o.get("P0_naive")
                o["LLM_NID"] = ref_res if routes["Ours"] == "fixed" else o.get("P0_naive")
                o["Oracle_NID"] = ref_res if r["intent"] == "fixed" else o.get("P0_naive")
                r["vlm"], r["n_nid"] = tag, len(nid)
                rel.append(r)
    ok = lambda m: (lambda r: diag.A.tgt(r["out"].get(m)) == r["gold"])
    names = [("P0", "Retry"), ("P0_naive", "Re-plan"), ("E", "Explicitize"), ("H", "Event history"),
             ("FREEZE", "Freeze ID"), ("BIND_OWN", "VLM-routed"), ("Ours", "BIND-LLM"), ("BIND_ID", "BIND-ID"),
             ("Oracle", "Oracle intent"), ("SR", "Surface rule"), ("SR_NID", "Surface rule + note id"),
             ("LLM_NID", "BIND-LLM + note id"), ("Oracle_NID", "Oracle intent + note id")]
    fx = [r for r in rel if r["intent"] == "fixed"]
    dy = [r for r in rel if r["intent"] == "dynamic"]
    print(f"units {len(rel)} (REF {len(fx)}, ATT {len(dy)}); REF units with exactly one noted target-type id: "
          f"{sum(r['n_nid'] == 1 for r in fx)}, ATT units with any noted target-type id: {sum(r['n_nid'] > 0 for r in dy)}")
    out = {}
    print(f"{'method':26s}{'all':>6s}{'REF':>6s}{'ATT':>6s}   diff vs BIND-LLM [95% CI]   per VLM")
    for m, nm in names:
        if not all(m in r["out"] for r in rel):
            continue
        acc = 100 * sum(map(ok(m), rel)) / len(rel)
        ra = 100 * sum(map(ok(m), fx)) / len(fx)
        aa = 100 * sum(map(ok(m), dy)) / len(dy)
        d, lo, hi = boot_diff(rel, ok(m), ok("Ours"))
        dr, lor, hir = boot_diff(fx, ok(m), ok("Ours"))
        da, loa, hia = boot_diff(dy, ok(m), ok("Ours"))
        per = {}
        for v in sorted({r["vlm"] for r in rel}):
            s = [r for r in rel if r["vlm"] == v]
            per[v] = round(100 * sum(map(ok(m), s)) / len(s), 1)
        out[nm] = dict(acc=acc, ref=ra, att=aa, diff=[d, lo, hi], diff_ref=[dr, lor, hir], diff_att=[da, loa, hia],
                       per_vlm=per)
        print(f"{nm:26s}{acc:6.1f}{ra:6.1f}{aa:6.1f}   {d:+5.1f} [{lo:+5.1f},{hi:+5.1f}]  "
              f"REF {dr:+5.1f} [{lor:+5.1f},{hir:+5.1f}] ATT {da:+5.1f} [{loa:+5.1f},{hia:+5.1f}]  {per}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
