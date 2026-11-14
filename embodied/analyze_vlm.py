"""Score EmbodiedDrift runs.  usage: python analyze_vlm.py res/both/qwen3vl8b.shard*.jsonl [--by family] [--json out]
drift  : fixed/rel units whose draft picked ref_B -> final target == ref_C   (retry-induced re-binding)
freeze : dynamic/rel units -> final target == ref_B                          (wrongly frozen)
named  : named/rel drift (control);  unr : accuracy on the unrelated-change retry (control)
CIs: scene-cluster bootstrap, 2000 reps, seed 2026.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

CFGS = ["draft", "P0_naive", "P0", "P0_noid", "P0_aware", "P0_draft", "H", "E", "P1", "FREEZE", "KW", "BIND", "BIND_E"]
MARK = __import__("re").compile(r"\b(whichever|whatever|at that point|when you get (there|to it)|at the time|by then|at that time)\b", __import__("re").I)


def tgt(op):
    return (op or {}).get("target")


def boot(groups, f, reps=2000, seed=2026):
    rng = random.Random(seed)
    ks = list(groups)
    vals = []
    for _ in range(reps):
        s = [u for k in (rng.choice(ks) for _ in ks) for u in groups[k]]
        v = f(s)
        if v is not None:
            vals.append(v)
    if not vals:
        return None, None
    vals.sort()
    return vals[int(.025 * len(vals))], vals[int(.975 * len(vals)) - 1]


def rate(units, pred):
    return None if not units else sum(map(pred, units)) / len(units)


def stat(units, pred):
    g = defaultdict(list)
    for u in units:
        g[u["episode_id"].split("_s")[0]].append(u)
    p = rate(units, pred)
    lo, hi = boot(g, lambda s: rate(s, pred)) if units else (None, None)
    return dict(p=p, lo=lo, hi=hi, n=len(units))


def summarize(R):
    S = {}
    sel = lambda i, c: [r for r in R if r["intent"] == i and r["cond"] == c]
    fx, dy, nm = sel("fixed", "rel"), sel("dynamic", "rel"), sel("named", "rel")
    fxu, dyu = sel("fixed", "unr"), sel("dynamic", "unr")
    S["init_bind"] = stat(fx, lambda r: tgt(r["out"]["draft"]) == r["ref_B"])
    S["kind_acc"] = stat(fx + dy, lambda r: r["kind"] == ("fixed" if r["intent"] == "fixed" else "dynamic"))
    for c in CFGS:
        fx0 = [r for r in fx if tgt(r["out"]["draft"]) == r["ref_B"]]
        nm0 = [r for r in nm if tgt(r["out"]["draft"]) == r["ref_B"]]
        S[f"drift/{c}"] = stat(fx0, lambda r, c=c: tgt(r["out"].get(c)) == r["ref_C"])
        S[f"named/{c}"] = stat(nm0, lambda r, c=c: tgt(r["out"].get(c)) == r["ref_C"])
        S[f"freeze/{c}"] = stat(dy, lambda r, c=c: tgt(r["out"].get(c)) == r["ref_B"])
        S[f"acc_rel/{c}"] = stat(fx + dy, lambda r, c=c: tgt(r["out"].get(c)) == r["gold"])
        S[f"acc_unr/{c}"] = stat(fxu + dyu, lambda r, c=c: tgt(r["out"].get(c)) == r["gold"])
    return S


def show(tag, S, n_eps):
    f = lambda z: "  n/a   " if z["p"] is None else f"{z['p']*100:3.0f}[{z['lo']*100:2.0f},{z['hi']*100:3.0f}]"
    print(f"\n### {tag}: {n_eps} episodes | init-bind {S['init_bind']['p']*100:.0f}% (n={S['init_bind']['n']}) "
          f"| kind acc {S['kind_acc']['p']*100:.0f}% | drift n={S['drift/P0']['n']} freeze n={S['freeze/P0']['n']}")
    show_c = CFGS[1:]
    print(f"{'':9s}" + "".join(f"{c:>13s}" for c in show_c))
    for row in ("drift", "freeze", "named"):
        print(f"{row:9s}" + "".join(f"{f(S[f'{row}/{c}']):>13s}" for c in show_c))
    for row in ("acc_rel", "acc_unr"):
        print(f"{row:9s}" + "".join(f"{S[f'{row}/{c}']['p']*100:13.0f}" if S[f'{row}/{c}']['p'] is not None else f"{'n/a':>13s}" for c in show_c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--by", choices=["family", "room"])
    ap.add_argument("--json")
    ap.add_argument("--instr", help="instructions jsonl, enables the KW keyword baseline")
    a = ap.parse_args()
    groups = defaultdict(list)
    instr = {}
    if a.instr:
        for l in open(a.instr, encoding="utf-8"):
            r = json.loads(l)
            instr[r["episode_id"]] = r
    import importlib.util
    spec = importlib.util.spec_from_file_location("rv", str(Path(__file__).with_name("run_vlm.py")))
    rv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rv)
    for fn in a.files:
        tag = Path(fn).name.split(".shard")[0]
        for l in open(fn):
            r = json.loads(l)
            o = r["out"]
            # re-parse every model call from raw text with the current parser (uniform across runs)
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
                r["mode"] = mode
                o["BIND"] = d if (mode == "referential" and d) else o.get("P0_naive")
                o["BIND_E"] = d if (mode == "referential" and d) else o.get("E")
            if instr and r["episode_id"] in instr and r["intent"] in instr[r["episode_id"]]:
                txt = instr[r["episode_id"]][r["intent"]]["instruction"]
                o["KW"] = o.get("P0_naive") if MARK.search(txt) else d
            groups[tag].append(r)
    if not instr:
        CFGS[:] = [c for c in CFGS if c != "KW"]
    allS = {}
    for tag, R in sorted(groups.items()):
        S = summarize(R)
        allS[tag] = S
        show(tag, S, len({r["episode_id"] for r in R}))
        if a.by:
            sub = defaultdict(list)
            for r in R:
                k = r[a.by]
                sub[k if a.by == "room" else ("lr" if k in ("left", "right") else k)].append(r)
            for k, Rs in sorted(sub.items()):
                S2 = summarize(Rs)
                allS[f"{tag}/{k}"] = S2
                show(f"{tag} [{a.by}={k}]", S2, len({r['episode_id'] for r in Rs}))
    if a.json:
        Path(a.json).write_text(json.dumps(allS, indent=1))


if __name__ == "__main__":
    main()
