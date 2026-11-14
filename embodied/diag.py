"""E1 failure taxonomy + E3 single- vs two-view re-resolution (CPU only, re-scores existing raw outputs).

usage: python diag.py --files res_grounded/both/*.jsonl --instr instr.jsonl [--intent modes.jsonl] --label grounded --json out.json
Every wrong final target on a rel unit (fixed + dynamic) gets ONE category, fixed precedence (declared before scoring):
  invalid   no parsable object id
  init      read-time draft != B-referent (initial grounding failure)
  route     routed method, route disagrees with gold intent (KW, self-tag, BIND own, Ours, Ours-2V)
  drift     referential request committed to the C-referent
  freeze    attributive request committed to the stale B-referent
  other     any other object (resolver error after a correct route)
Named-control drift (switch of an explicitly named object) is reported separately as the visual-correspondence probe.
"""
import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import analyze_vlm as A  # noqa: E402

CATS = ["invalid", "init", "route", "drift", "freeze", "other"]
OBJ = re.compile(r"^obj\d+$")


def load(files, instr, ext, rv):
    groups = defaultdict(list)
    for fn in files:
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
            r["route"] = {}
            o["FREEZE"] = d
            o["P1"] = d if (r.get("kind") == "fixed" and d) else o.get("P0")
            r["route"]["P1"] = "fixed" if r.get("kind") == "fixed" else "dynamic"
            if r.get("mode_raw") is not None:
                m = rv.parse_mode(r["mode_raw"])
                o["BIND_OWN"] = d if (m == "referential" and d) else o.get("P0_naive")
                r["route"]["BIND_OWN"] = "fixed" if m == "referential" else "dynamic"
            if ext is not None:
                p = ext.get((r["episode_id"], r["intent"]))
                o["Ours"] = d if (p == "fixed" and d) else o.get("P0_naive")
                o["Ours_2V"] = d if (p == "fixed" and d) else o.get("P0")
                o["Oracle"] = d if (r["intent"] in ("fixed", "named") and d) else o.get("P0_naive")
                for k in ("Ours", "Ours_2V"):
                    r["route"][k] = "fixed" if p == "fixed" else "dynamic"
            if instr and r["episode_id"] in instr and r["intent"] in instr[r["episode_id"]]:
                txt = instr[r["episode_id"]][r["intent"]]["instruction"]
                kw_dyn = bool(A.MARK.search(txt))
                o["KW"] = o.get("P0_naive") if kw_dyn else d
                r["route"]["KW"] = "dynamic" if kw_dyn else "fixed"
            groups[tag].append(r)
    return groups


def category(r, m):
    t = A.tgt(r["out"].get(m))
    if t == r["gold"]:
        return "correct"
    if not t or not OBJ.match(str(t)):
        return "invalid"
    if A.tgt(r["out"].get("draft")) != r["ref_B"]:
        return "init"
    rt = r["route"].get(m)
    if rt is not None and rt != r["intent"]:
        return "route"
    if r["intent"] == "fixed" and t == r["ref_C"]:
        return "drift"
    if r["intent"] == "dynamic" and t == r["ref_B"]:
        return "freeze"
    return "other"


def boot_ci(units, f, reps=2000, seed=2026):
    g = defaultdict(list)
    for u in units:
        g[u["episode_id"].split("_s")[0]].append(u)
    ks, rng, vals = list(g), random.Random(seed), []
    for _ in range(reps):
        s = [u for k in (rng.choice(ks) for _ in ks) for u in g[k]]
        vals.append(sum(map(f, s)) / len(s))
    vals.sort()
    return vals[int(.025 * reps)], vals[int(.975 * reps) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--instr")
    ap.add_argument("--intent")
    ap.add_argument("--label", required=True)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
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
    methods = ["P0_naive", "P0", "P0_aware", "H", "E", "P1", "FREEZE", "KW", "BIND_OWN", "Ours", "Ours_2V", "Oracle"]
    out = {}
    for tag, R in sorted(load(a.files, instr, ext, rv).items()):
        rel = [r for r in R if r["cond"] == "rel" and r["intent"] in ("fixed", "dynamic")]
        nm = [r for r in R if r["cond"] == "rel" and r["intent"] == "named"
              and A.tgt(r["out"].get("draft")) == r["ref_B"]]
        unr = [r for r in R if r["cond"] == "unr" and r["intent"] in ("fixed", "dynamic")]
        res = {"n_rel": len(rel), "n_named": len(nm), "n_unr": len(unr)}
        print(f"\n### {a.label} / {tag}: rel units {len(rel)} (fixed {sum(r['intent']=='fixed' for r in rel)}), "
              f"named(draft ok) {len(nm)}, unr {len(unr)}")
        print(f"{'method':10s}{'acc':>6s}" + "".join(f"{c:>8s}" for c in CATS) + f"{'named':>8s}{'unr':>6s}")
        for m in methods:
            if not any(m in r["out"] for r in rel):
                continue
            cnt = Counter(category(r, m) for r in rel)
            byint = {i: Counter(category(r, m) for r in rel if r["intent"] == i) for i in ("fixed", "dynamic")}
            ndrift = sum(A.tgt(r["out"].get(m)) == r["ref_C"] for r in nm)
            uacc = sum(A.tgt(r["out"].get(m)) == r["gold"] for r in unr)
            lo, hi = boot_ci(rel, lambda r, m=m: A.tgt(r["out"].get(m)) == r["gold"])
            res[m] = {"counts": dict(cnt), "by_intent": {k: dict(v) for k, v in byint.items()},
                      "acc": cnt["correct"] / len(rel), "acc_ci": [lo, hi],
                      "named_drift": ndrift / max(1, len(nm)), "named_n": len(nm),
                      "unr_acc": uacc / max(1, len(unr)), "unr_n": len(unr)}
            print(f"{m:10s}{100*cnt['correct']/len(rel):6.1f}" + "".join(f"{cnt[c]:8d}" for c in CATS)
                  + f"{100*ndrift/max(1,len(nm)):8.1f}{100*uacc/max(1,len(unr)):6.1f}")
        out[tag] = res
    Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
