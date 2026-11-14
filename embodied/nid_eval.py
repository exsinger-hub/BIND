"""Semantic note-id router (mode_classify --variant noteid; review v7 exp #5): the router names the related note's bound
object id; REF commits that id when it is an object of the target type, else the draft; ATT re-resolves (Re-plan).
Compared with BIND-LLM (same router family, draft replay) and the construction-aware Surface rule + note id.
usage: python nid_eval.py --files <raw> --instr <instr> --episodes episodes_v1 --llm <modes> --nid <noteid modes>
"""
import argparse, importlib.util, json, re
from pathlib import Path
import diag
from controls import ID, r4
from review_stats import boot_diff

ap = argparse.ArgumentParser()
for k in ("--instr", "--episodes", "--llm", "--nid"):
    ap.add_argument(k, required=True)
ap.add_argument("--files", nargs="+", required=True)
a = ap.parse_args()
spec = importlib.util.spec_from_file_location("rv", str(Path(diag.__file__).with_name("run_vlm.py")))
rv = importlib.util.module_from_spec(spec); spec.loader.exec_module(rv)
instr = {json.loads(l)["episode_id"]: json.loads(l) for l in open(a.instr, encoding="utf-8")}
llm = {(m["episode_id"], m["intent"]): m["pred"] for m in map(json.loads, open(a.llm))}
nid = {(m["episode_id"], m["intent"]): m for m in map(json.loads, open(a.nid))}
rel, eps = [], {}
for f in a.files:
    for tag, R in diag.load([f], instr, llm, rv).items():
        for r in R:
            if r["cond"] != "rel" or r["intent"] not in ("fixed", "dynamic"):
                continue
            eid = r["episode_id"]
            ep = eps.setdefault(eid, json.loads((Path(a.episodes) / eid / "episode.json").read_text()))
            typed = {x["id"] for x in ep["records_B"] if x["type"] == ep["target_type"]}
            m = nid.get((eid, r["intent"]), {})
            d = r["out"].get("draft")
            b = m.get("bid")
            r["out"]["NID"] = ({"target": b} if b in typed else d) if m.get("pred") == "fixed" else r["out"].get("P0_naive")
            notes = instr[eid][r["intent"]]["notes"]
            noun = re.sub(r"(?<!^)(?=[A-Z])", " ", ep["target_type"]).lower()
            ids = sorted({i for n in notes for i in ID.findall(n) if i in typed})
            r["out"]["SRNID"] = ({"target": ids[0]} if len(ids) == 1 else d) if r4(notes, noun) == "fixed" else r["out"].get("P0_naive")
            r["nid_route_ok"] = m.get("pred") == r["intent"]
            rel.append(r)
ok = lambda k: (lambda r: diag.A.tgt(r["out"].get(k)) == r["gold"])
fx = [r for r in rel if r["intent"] == "fixed"]; dy = [r for r in rel if r["intent"] == "dynamic"]
print(f"units {len(rel)}; noteid-router routing acc {100*sum(r['nid_route_ok'] for r in rel)/len(rel):.1f}")
for k, n in (("Ours", "BIND-LLM (draft replay)"), ("NID", "BIND-LLM + semantic note id"), ("SRNID", "Surface rule + note id"), ("Oracle", "Oracle intent")):
    dd, lo, hi = boot_diff(rel, ok(k), ok("Ours"))
    print(f"{n:30s} all {100*sum(map(ok(k), rel))/len(rel):5.1f} REF {100*sum(map(ok(k), fx))/len(fx):5.1f} ATT {100*sum(map(ok(k), dy))/len(dy):5.1f}  vs BIND-LLM {dd:+.1f} [{lo:+.1f},{hi:+.1f}]")
