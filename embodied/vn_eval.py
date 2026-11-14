"""Visual-necessity controls (v6 review exp #4 / nov #10), 201-episode subset of the memory-grounded split.
Per memory condition (both = images + records, img = images only, graph = records only, shuf = records + images of
another scene): read-time draft accuracy, Retry and Re-plan accuracy, BIND-LLM and oracle-intent accuracy (REF/ATT).
usage: python vn_eval.py --instr instructions_grounded.jsonl --llm modes_dg/qwen3.5-9b_grounded_struct.jsonl
       --runs both=res_e7/both/q3vl_v0.shard0of1.jsonl img=res_x_vn/img/q3vl_bf16.shard0of1.jsonl ...
"""
import argparse
import importlib.util
import json
from pathlib import Path

import diag
from review_stats import boot_diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instr", required=True)
    ap.add_argument("--llm", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("rv", str(Path(diag.__file__).with_name("run_vlm.py")))
    rv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rv)
    instr = {json.loads(l)["episode_id"]: json.loads(l) for l in open(a.instr, encoding="utf-8")}
    llm = {(m["episode_id"], m["intent"]): m["pred"] for m in map(json.loads, open(a.llm))}
    ok = lambda m: (lambda r: diag.A.tgt(r["out"].get(m)) == r["gold"])
    draft_ok = lambda r: diag.A.tgt(r["out"].get("draft")) == r["ref_B"]
    runs = {}
    for s in a.runs:
        n, f = s.split("=", 1)
        rel = [r for R in diag.load([f], instr, llm, rv).values() for r in R
               if r["cond"] == "rel" and r["intent"] in ("fixed", "dynamic")]
        runs[n] = {(r["episode_id"], r["intent"]): r for r in rel}
    common = set.intersection(*(set(v) for v in runs.values()))
    print(f"common units {len(common)}")
    base = list(runs)[0]
    for n, R in runs.items():
        rel = [R[k] for k in sorted(common)]
        fx = [r for r in rel if r["intent"] == "fixed"]
        dy = [r for r in rel if r["intent"] == "dynamic"]
        pc = lambda f, s: 100 * sum(map(f, s)) / len(s)
        line = (f"{n:6s} draft {pc(draft_ok, rel):5.1f} | Retry REF {pc(ok('P0'), fx):5.1f} ATT {pc(ok('P0'), dy):5.1f} | "
                f"Re-plan ATT {pc(ok('P0_naive'), dy):5.1f} | BIND-LLM all {pc(ok('Ours'), rel):5.1f} REF {pc(ok('Ours'), fx):5.1f} "
                f"ATT {pc(ok('Ours'), dy):5.1f} | Oracle {pc(ok('Oracle'), rel):5.1f}")
        if n != base:
            pair = [(runs[base][k], R[k]) for k in sorted(common)]
            # paired difference of BIND-LLM accuracy vs the base condition (cluster bootstrap on scenes)
            units = [dict(episode_id=k[0], a=ok("Ours")(R[k]), b=ok("Ours")(runs[base][k])) for k in sorted(common)]
            d, lo, hi = boot_diff(units, lambda u: u["a"], lambda u: u["b"])
            line += f" | BIND-LLM vs {base} {d:+.1f} [{lo:+.1f},{hi:+.1f}]"
        print(line)


if __name__ == "__main__":
    main()
