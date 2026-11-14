"""Prompt-matched controls (v6 review exp #3 / nov #4) and the image-B ablation (exp #9), memory-grounded split.
On one grounding run (same checkpoint): REF/ATT/all accuracy of
  P0 (retry), P0_noB (retry without the version-1 image), P0_naive (re-plan), P0_struct (single-stage resolver given
  the BIND-LLM procedure), BIND-LLM (Qwen3.5-9B router), Self-struct (the grounding VLM itself as structured text router),
  Oracle intent; paired scene-cluster CIs vs BIND-LLM.
usage: python pm_eval.py --files res_x_pm/both/TAG.shard0of1.jsonl --instr instructions_grounded.jsonl
       --llm modes_dg/qwen3.5-9b_grounded_struct.jsonl --self modes_pm/TAG_grounded_struct.jsonl
"""
import argparse
import importlib.util
import json
from pathlib import Path

import diag
from review_stats import boot_diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--instr", required=True)
    ap.add_argument("--llm", required=True)
    ap.add_argument("--self", dest="self_", required=True)
    ap.add_argument("--extra", nargs="*", default=[], help="other runs of the same checkpoint whose stage-2 configs are merged by unit key (drafts must agree)")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("rv", str(Path(diag.__file__).with_name("run_vlm.py")))
    rv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rv)
    instr = {json.loads(l)["episode_id"]: json.loads(l) for l in open(a.instr, encoding="utf-8")}
    llm = {(m["episode_id"], m["intent"]): m["pred"] for m in map(json.loads, open(a.llm))}
    slf = {(m["episode_id"], m["intent"]): m["pred"] for m in map(json.loads, open(a.self_))}
    extra, dmis = {}, 0
    for f in a.extra:
        for R in diag.load([f], instr, llm, rv).values():
            for r in R:
                extra[r["key"]] = r
    rel = []
    for f in a.files:
        for tag, R in diag.load([f], instr, llm, rv).items():
            for r in R:
                if r["cond"] == "rel" and r["intent"] in ("fixed", "dynamic"):
                    e = extra.get(r["key"])
                    if e is not None:
                        dmis += diag.A.tgt(e["out"].get("draft")) != diag.A.tgt(r["out"].get("draft"))
                        for c in ("P0_struct", "P0_structd", "P0_guard", "P0_noB"):
                            if c in e["out"]:
                                r["out"][c] = e["out"][c]
                    o, d = r["out"], r["out"].get("draft")
                    p = slf.get((r["episode_id"], r["intent"]))
                    o["Self_struct"] = d if p == "fixed" and d else o.get("P0_naive")
                    r["self_route_ok"] = p == r["intent"]
                    r["llm_route_ok"] = llm.get((r["episode_id"], r["intent"])) == r["intent"]
                    rel.append(r)
    ok = lambda m: (lambda r: diag.A.tgt(r["out"].get(m)) == r["gold"])
    fx = [r for r in rel if r["intent"] == "fixed"]
    dy = [r for r in rel if r["intent"] == "dynamic"]
    print(f"draft mismatches with merged runs: {dmis}")
    print(f"units {len(rel)}; routing acc BIND-LLM {100*sum(r['llm_route_ok'] for r in rel)/len(rel):.1f}, "
          f"self-struct {100*sum(r['self_route_ok'] for r in rel)/len(rel):.1f}")
    for m, nm in (("P0", "Retry"), ("P0_noB", "Retry w/o image B"), ("P0_naive", "Re-plan"),
                  ("P0_struct", "Single-stage structured"), ("P0_structd", "Draft-aware single call"), ("P0_guard", "Guarded retry"), ("Ours", "BIND-LLM"), ("Self_struct", "Self-structured router"),
                  ("Ours_2V", "BIND-LLM two-view"), ("Oracle", "Oracle intent")):
        if not any(m in r["out"] for r in rel):
            continue
        d, lo, hi = boot_diff(rel, ok(m), ok("Ours"))
        print(f"{nm:26s} all {100*sum(map(ok(m), rel))/len(rel):5.1f}  REF {100*sum(map(ok(m), fx))/len(fx):5.1f}  "
              f"ATT {100*sum(map(ok(m), dy))/len(dy):5.1f}   vs BIND-LLM {d:+5.1f} [{lo:+5.1f},{hi:+5.1f}]")
    d, lo, hi = boot_diff(dy, ok("P0_noB"), ok("P0"))
    print(f"ATT: retry w/o image B - retry with image B: {d:+.1f} [{lo:+.1f},{hi:+.1f}]")


if __name__ == "__main__":
    main()
