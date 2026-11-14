"""Score pilot outputs.  usage: python analyze.py results/*.jsonl [--keep keep_ids.txt] [--json out.json]

Primary per-unit outcome: referent correctness of the final op (target or value slot == gold).
Diagnostics (rel condition):
  fixed   : drift  = P(final slot == ref_C | draft slot == ref_B)      (retry-induced binding drift)
  dynamic : freeze = P(final slot == ref_B)                              (wrong freezing)
CIs: scenario-cluster bootstrap (2000 reps, seed 2026).
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

CONFIGS = ["draft", "P0_naive", "P0", "P0_draft", "P0_aware", "H", "E", "P1", "C1", "P1C1"]


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def slot_of(op, slot):
    if not op:
        return None
    return op["target"] if slot == "target" else op["value"]


def ref_ok(op, r):
    v = slot_of(op, r["slot"])
    return v is not None and norm(v) == norm(slot_of(r["gold"], r["slot"]))


def action_ok(op, r):
    if not ref_ok(op, r):
        return False
    g = r["gold"]
    if norm(op["target"]) != norm(g["target"]):
        return False
    fo, fg = norm(op["field"]), norm(g["field"])
    return fo == fg or (fo and fg and (fo in fg or fg in fo))


def boot(units_by_sc, stat, reps=2000, seed=2026):
    rng = random.Random(seed)
    keys = list(units_by_sc)
    vals = []
    for _ in range(reps):
        samp = [u for k in (rng.choice(keys) for _ in keys) for u in units_by_sc[k]]
        v = stat(samp)
        if v is not None:
            vals.append(v)
    if not vals:
        return (None, None)
    vals.sort()
    return (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1])


def rate(units, f, cond=lambda u: True):
    sel = [u for u in units if cond(u)]
    if not sel:
        return None
    return sum(f(u) for u in sel) / len(sel)


def summarize(recs):
    out = {}
    by_sc = defaultdict(list)
    for r in recs:
        by_sc[r["scenario_id"]].append(r)

    def grp(name, cond, f, denom=None):
        sel = [r for r in recs if cond(r) and (denom is None or denom(r))]
        sc = defaultdict(list)
        for r in sel:
            sc[r["scenario_id"]].append(r)
        p = rate(sel, f)
        lo, hi = boot(sc, lambda s: rate(s, f)) if sel else (None, None)
        out[name] = dict(p=p, lo=lo, hi=hi, n=len(sel), n_sc=len(sc))

    for cfg in CONFIGS:
        for t in ("fixed", "dynamic"):
            for cn in ("rel", "unrel", "named"):
                if t == "dynamic" and cn == "named":
                    continue
                grp(f"acc/{cfg}/{t}/{cn}", lambda r, t=t, cn=cn: r["referent_type"] == t and r["cond"] == cn,
                    lambda r, cfg=cfg: ref_ok(r["out"].get(cfg), r))
        # retry-induced drift among initially-correct fixed rel units
        grp(f"drift/{cfg}/fixed/rel", lambda r: r["referent_type"] == "fixed" and r["cond"] == "rel",
            lambda r, cfg=cfg: norm(slot_of(r["out"].get(cfg), r["slot"]) or "") == norm(r["ref_C"]),
            denom=lambda r: norm(slot_of(r["draft"], r["slot"]) or "") == norm(r["ref_B"]))
        grp(f"drift/{cfg}/fixed/named", lambda r: r["referent_type"] == "fixed" and r["cond"] == "named",
            lambda r, cfg=cfg: norm(slot_of(r["out"].get(cfg), r["slot"]) or "") == norm(r["ref_C"]),
            denom=lambda r: norm(slot_of(r["draft"], r["slot"]) or "") == norm(r["ref_B"]))
        grp(f"freeze/{cfg}/dynamic/rel", lambda r: r["referent_type"] == "dynamic" and r["cond"] == "rel",
            lambda r, cfg=cfg: norm(slot_of(r["out"].get(cfg), r["slot"]) or "") == norm(r["ref_B"]))
        grp(f"action/{cfg}/all/rel", lambda r: r["cond"] == "rel", lambda r, cfg=cfg: action_ok(r["out"].get(cfg), r))
    grp("initial_binding_acc/fixed/rel", lambda r: r["referent_type"] == "fixed" and r["cond"] == "rel",
        lambda r: norm(slot_of(r["draft"], r["slot"]) or "") == norm(r["ref_B"]))
    grp("kind_acc/all/rel", lambda r: r["cond"] == "rel", lambda r: r["draft_kind"] == r["referent_type"])
    grp("merge_conflict/all/rel", lambda r: r["cond"] == "rel", lambda r: bool(r["merge_conflict"]))
    grp("parse_fail/P0/all", lambda r: True, lambda r: r["out"].get("P0") is None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--keep", help="file with scenario ids to keep (main set)")
    ap.add_argument("--json")
    args = ap.parse_args()
    keep = set(Path(args.keep).read_text().split()) if args.keep else None
    allres = {}
    for fn in args.files:
        recs = [json.loads(l) for l in open(fn, encoding="utf-8")]
        if keep is not None:
            recs = [r for r in recs if r["scenario_id"] in keep]
        tag = Path(fn).stem
        s = summarize(recs)
        allres[tag] = s
        print(f"\n===== {tag}  ({len({r['scenario_id'] for r in recs})} scenarios)")
        ib = s["initial_binding_acc/fixed/rel"]
        print(f"initial binding acc (fixed, rel): {ib['p']:.3f}  kind acc: {s['kind_acc/all/rel']['p']:.3f}  "
              f"merge conflicts: {s['merge_conflict/all/rel']['p']:.3f}  P0 parse fail: {s['parse_fail/P0/all']['p']:.3f}")
        hdr = f"{'config':10s} {'fix/rel':>8s} {'fix/unr':>8s} {'fix/nam':>8s} {'dyn/rel':>8s} {'dyn/unr':>8s} | {'drift':>16s} {'drift/nam':>9s} {'freeze':>16s} {'action':>7s}"
        print(hdr)
        for cfg in CONFIGS:
            a = lambda k: s[k]["p"]
            d = s[f"drift/{cfg}/fixed/rel"]
            fz = s[f"freeze/{cfg}/dynamic/rel"]
            fmt = lambda x: f"{x:8.3f}" if x is not None else "     n/a"
            ci = lambda z: f"{z['p']:.2f}[{z['lo']:.2f},{z['hi']:.2f}]" if z["p"] is not None else "n/a"
            print(f"{cfg:10s} {fmt(a(f'acc/{cfg}/fixed/rel'))} {fmt(a(f'acc/{cfg}/fixed/unrel'))} "
                  f"{fmt(a(f'acc/{cfg}/fixed/named'))} {fmt(a(f'acc/{cfg}/dynamic/rel'))} "
                  f"{fmt(a(f'acc/{cfg}/dynamic/unrel'))} | {ci(d):>16s} {fmt(s[f'drift/{cfg}/fixed/named']['p']):>9s} "
                  f"{ci(fz):>16s} {fmt(a(f'action/{cfg}/all/rel')):>7s}")
        print(f"(drift n={s['drift/P0/fixed/rel']['n']} units / {s['drift/P0/fixed/rel']['n_sc']} scenarios; "
              f"freeze n={s['freeze/P0/dynamic/rel']['n']})")
    if args.json:
        Path(args.json).write_text(json.dumps(allres, indent=1))


if __name__ == "__main__":
    main()
