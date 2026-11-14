"""Binding-drift pilot runner (text). One frozen model, all configurations.

Per scenario we build three conditions:
  rel    : original instruction U, concurrent change that alters what the expression denotes
  unrel  : original instruction U, concurrent change that leaves the denotation unchanged
  named  : U with the expression replaced by an explicit name, concurrent change = rel
All configurations share ONE initial draft per (scenario, condition) produced on snapshot B.

Configurations evaluated on the post-conflict state C:
  draft      : the initial draft itself (what would have been committed without a conflict)
  P0_naive   : fresh extraction against C only (common practice: re-run the memory-update step)
  P0         : reinterpret with B, changelog, C (no draft)
  P0_draft   : P0 + the rejected draft
  P0_aware   : P0 + an explicit note to decide fixed vs re-evaluated reading (prompt-only fix)
  H          : full event-history decision (event log framing)
  E          : one-pass explicitization on B, then fresh extraction of the rewritten request on C
  P1         : keep draft if the model tagged its binding 'fixed', else P0
  C1         : field-level 3-way merge of the draft patch; model (P0_draft prompt) only on same-field conflict
  P1C1       : C1, but on conflict keep the draft if tagged 'fixed'

usage: python run_pilot.py --model Qwen/Qwen3-8B --tag qwen3-8b --data data/all --out results
"""
import argparse
import copy
import json
import re
import time
from pathlib import Path

SYSTEM = ("You are a memory-writer agent for a shared team memory that several agents read and write. "
          "You turn a user's request into a precise update of one memory record. Answer with JSON only.")

OP_SPEC = ('{"op": {"target": "<record id>", "field": "<field name>", '
           '"value": "<new value; if the value refers to a record, use its record id>"}}')

DRAFT_SPEC = ('{"op": {"target": "<record id>", "field": "<field name>", '
              '"value": "<new value; if the value refers to a record, use its record id>"}, '
              '"referring_expression": "<phrase in the request that identifies the record being updated or the record used as the value>", '
              '"resolved_id": "<record id that phrase identifies>", '
              '"kind": "fixed or dynamic"}')


# ----------------------------------------------------------------------------- state helpers
def apply_change(mem, change):
    mem = copy.deepcopy(mem)
    by_id = {r["id"]: r for r in mem}
    for r in change.get("new_records", []):
        mem.append(copy.deepcopy(r))
        by_id[r["id"]] = mem[-1]
    for p in change.get("patch", []):
        by_id[p["id"]]["fields"][p["field"]] = p["value"]
    return mem


def render_mem(mem):
    return "\n".join(json.dumps({"id": r["id"], "type": r["type"], **r["fields"]}, ensure_ascii=False) for r in mem)


def render_conv(context, instruction):
    lines = [f'{t["speaker"]}: {t["text"]}' for t in context]
    lines.append(f"user: {instruction}")
    return "\n".join(lines)


def render_change(change):
    parts = [f'{change["by"]}: "{change["utterance"]}"']
    for p in change.get("patch", []):
        parts.append(f'  set {p["id"]}.{p["field"]} = {json.dumps(p["value"], ensure_ascii=False)}')
    for r in change.get("new_records", []):
        parts.append("  add " + json.dumps({"id": r["id"], "type": r["type"], **r["fields"]}, ensure_ascii=False))
    return "\n".join(parts)


def conditions(sc):
    rel, unrel = sc["concurrent_relevant"], sc["concurrent_unrelated"]
    res, slot, g = sc["resolution"], sc["slot"], sc["gold_op"]

    def gold_with(ref):
        gg = dict(g)
        gg["target" if slot == "target" else "value"] = ref
        return gg

    out = {
        "rel": dict(instruction=sc["instruction"], change=rel, gold=gold_with(res["intended"]),
                    ref_B=res["in_B"], ref_C=res["in_C"]),
        # under the unrelated change the denotation is unchanged, so the intended referent is in_B for both types
        "unrel": dict(instruction=sc["instruction"], change=unrel, gold=gold_with(res["in_B"]),
                      ref_B=res["in_B"], ref_C=res["in_B"]),
    }
    if sc["referent_type"] == "fixed":
        out["named"] = dict(instruction=sc["named_instruction"], change=rel, gold=gold_with(res["intended"]),
                            ref_B=res["in_B"], ref_C=res["in_C"])
    return out


# ----------------------------------------------------------------------------- prompts
def p_draft(mem_B, context, instr):
    return (f"SHARED MEMORY (current version):\n{render_mem(mem_B)}\n\n"
            f"CONVERSATION WITH YOUR USER:\n{render_conv(context, instr)}\n\n"
            "Translate the user's latest request into a single field update on one memory record. "
            "Use an existing field name when one fits.\n"
            '"kind" is "fixed" if the phrase is meant to pick out the specific record it identifies right now, '
            'or "dynamic" if it is meant to be re-evaluated against the memory at the moment the update is applied.\n'
            f"Return only JSON in this format:\n{DRAFT_SPEC}")


def p_naive(mem_C, context, instr):
    return (f"SHARED MEMORY (current version):\n{render_mem(mem_C)}\n\n"
            f"CONVERSATION WITH YOUR USER:\n{render_conv(context, instr)}\n\n"
            "Translate the user's latest request into a single field update on one memory record. "
            "Use an existing field name when one fits.\n"
            f"Return only JSON in this format:\n{OP_SPEC}")


AWARE_NOTE = ("Important: the user spoke while looking at version 1. For every description in the request that "
              "identifies a record (a role, an ordering such as 'latest', a property, a pronoun, a time), decide what "
              "the user meant: the specific record the description picked out in version 1, or whatever the description "
              "picks out at the moment the update is applied. Use the conversation to decide, and carry out what the "
              "user meant.\n\n")


def p_retry(mem_B, change, mem_C, context, instr, draft=None, aware=False):
    s = ("Your write to the shared memory was rejected by the version check: while you were drafting it, "
         "another agent committed a change. Re-apply the user's request against the CURRENT memory.\n\n"
         + (AWARE_NOTE if aware else "") +
         f"MEMORY YOU READ WHEN THE USER SPOKE (version 1):\n{render_mem(mem_B)}\n\n"
         f"CHANGE COMMITTED SINCE THEN (version 2):\n{render_change(change)}\n\n"
         f"CURRENT MEMORY (version 2):\n{render_mem(mem_C)}\n\n"
         f"CONVERSATION WITH YOUR USER:\n{render_conv(context, instr)}\n\n")
    if draft is not None:
        s += f"YOUR REJECTED DRAFT (written against version 1):\n{json.dumps({'op': draft}, ensure_ascii=False)}\n\n"
    s += ("Produce the single field update to commit now. Use an existing field name when one fits.\n"
          f"Return only JSON in this format:\n{OP_SPEC}")
    return s


def p_history(mem_B, change, writer, other, context, instr):
    return ("EVENT LOG OF THE SHARED MEMORY\n\n"
            f"[v1] memory state:\n{render_mem(mem_B)}\n\n"
            f"[event A] {other} (had read v1) committed v2:\n{render_change(change)}\n\n"
            f"[event B] {writer} (had read v1) received this conversation from its user and requested an update, "
            f"committed after event A:\n{render_conv(context, instr)}\n\n"
            "Determine the single field update that correctly carries out the request in event B, applied to the "
            "memory after event A. Use an existing field name when one fits.\n"
            f"Return only JSON in this format:\n{OP_SPEC}")


def p_explicit(mem_B, context, instr):
    return (f"SHARED MEMORY (current version):\n{render_mem(mem_B)}\n\n"
            f"CONVERSATION WITH YOUR USER:\n{render_conv(context, instr)}\n\n"
            "Rewrite the user's latest request as a standalone instruction that another agent can execute later "
            "without this conversation. Replace pronouns and descriptions that refer to specific records with the "
            "explicit names (and ids) of those records. If the user means a description to be evaluated when the "
            "instruction is executed (for example 'whoever is on call at that point'), keep that description as is. "
            "Keep everything else unchanged.\n"
            'Return only JSON in this format:\n{"rewritten": "<standalone instruction>"}')


# ----------------------------------------------------------------------------- parsing
def parse_json(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    s = m.group(0)
    for cand in (s, s[: s.rfind("}") + 1]):
        try:
            return json.loads(cand)
        except Exception:  # noqa: BLE001
            pass
    # fall back: first balanced object
    depth = 0
    for i, ch in enumerate(s):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            try:
                return json.loads(s[: i + 1])
            except Exception:  # noqa: BLE001
                return None
    return None


def get_op(obj):
    if not isinstance(obj, dict):
        return None
    op = obj.get("op", obj)
    if not isinstance(op, dict) or "target" not in op:
        return None
    return {k: str(op.get(k, "")) for k in ("target", "field", "value")}


# ----------------------------------------------------------------------------- engine
class Engine:
    def __init__(self, model, max_len, gpu_util, tp, seed):
        from vllm import LLM, SamplingParams
        kw = dict(model=model, max_model_len=max_len, gpu_memory_utilization=gpu_util, tensor_parallel_size=tp,
                  seed=seed, trust_remote_code=True, enable_prefix_caching=True, enforce_eager=True)
        if "mistral" in model.lower():
            kw.update(tokenizer_mode="mistral", config_format="mistral", load_format="mistral")
        try:
            kw_mm = dict(kw, limit_mm_per_prompt={"image": 0, "video": 0})
            self.llm = LLM(**kw_mm)
        except Exception:  # noqa: BLE001
            self.llm = LLM(**kw)
        self.sp = SamplingParams(temperature=0.0, max_tokens=512, seed=seed)
        self.model = model
        self.calls = 0
        self.in_tok = 0
        self.out_tok = 0

    def run(self, prompts):
        if not prompts:
            return []
        convs = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}] for p in prompts]
        kw = {}
        if "qwen" in self.model.lower():
            kw["chat_template_kwargs"] = {"enable_thinking": False}
        outs = self.llm.chat(convs, self.sp, use_tqdm=True, **kw)
        res = []
        for o in outs:
            self.calls += 1
            self.in_tok += len(o.prompt_token_ids)
            self.out_tok += len(o.outputs[0].token_ids)
            res.append(o.outputs[0].text)
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--gpu-util", type=float, default=0.80)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    scs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(args.data).glob("*.json"))]
    units = []  # (scenario, cond_name, cond)
    for sc in scs:
        for cn, c in conditions(sc).items():
            units.append((sc, cn, c))
    print(f"{len(scs)} scenarios, {len(units)} units", flush=True)

    eng = Engine(args.model, args.max_len, args.gpu_util, args.tp, args.seed)
    t0 = time.time()

    # stage 1: initial draft on B (shared by all configs) + explicitization rewrite on B
    s1 = [p_draft(sc["memory_B"], sc["context"], c["instruction"]) for sc, _, c in units]
    s1e = [p_explicit(sc["memory_B"], sc["context"], c["instruction"]) for sc, _, c in units]
    r1 = eng.run(s1 + s1e)
    drafts_raw, expl_raw = r1[: len(units)], r1[len(units):]

    recs = []
    for (sc, cn, c), dr, er in zip(units, drafts_raw, expl_raw):
        dobj = parse_json(dr) or {}
        eobj = parse_json(er) or {}
        mem_C = apply_change(sc["memory_B"], c["change"])
        recs.append(dict(scenario_id=sc["scenario_id"], author=sc["author"], cond=cn,
                         referent_type=sc["referent_type"], expr_category=sc["expr_category"], cue=sc["cue"],
                         slot=sc["slot"], gold=c["gold"], ref_B=c["ref_B"], ref_C=c["ref_C"],
                         draft=get_op(dobj), draft_kind=str(dobj.get("kind", "")).lower().strip(),
                         draft_raw=dr, rewritten=eobj.get("rewritten") if isinstance(eobj, dict) else None,
                         expl_raw=er, _mem_C=mem_C, _sc=sc, _c=c, out={}, raw={}))

    # stage 2: retry prompts on C
    jobs = []  # (rec_idx, config, prompt)
    for i, r in enumerate(recs):
        sc, c, mem_C = r["_sc"], r["_c"], r["_mem_C"]
        mB, ctx, ins, ch = sc["memory_B"], sc["context"], c["instruction"], c["change"]
        jobs.append((i, "P0_naive", p_naive(mem_C, ctx, ins)))
        jobs.append((i, "P0", p_retry(mB, ch, mem_C, ctx, ins)))
        jobs.append((i, "P0_draft", p_retry(mB, ch, mem_C, ctx, ins, draft=r["draft"])))
        jobs.append((i, "P0_aware", p_retry(mB, ch, mem_C, ctx, ins, aware=True)))
        jobs.append((i, "H", p_history(mB, ch, sc["writers"]["writer"], sc["writers"]["other"], ctx, ins)))
        rw = r["rewritten"] if isinstance(r["rewritten"], str) and r["rewritten"].strip() else ins
        jobs.append((i, "E", p_naive(mem_C, [], rw)))
    r2 = eng.run([j[2] for j in jobs])
    for (i, cfg, _), txt in zip(jobs, r2):
        recs[i]["out"][cfg] = get_op(parse_json(txt))
        recs[i]["raw"][cfg] = txt

    # deterministic configs
    for r in recs:
        d = r["draft"]
        r["out"]["draft"] = d
        fixed = r["draft_kind"] == "fixed"
        r["out"]["P1"] = d if (fixed and d) else r["out"]["P0"]
        touched = {(p["id"], p["field"]) for p in r["_c"]["change"].get("patch", [])}
        conflict = d is None or (d["target"], d["field"]) in touched
        r["merge_conflict"] = conflict
        r["out"]["C1"] = r["out"]["P0_draft"] if conflict else d
        r["out"]["P1C1"] = (d if (fixed and d) else r["out"]["P0_draft"]) if conflict else d

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{args.tag}.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            r = {k: v for k, v in r.items() if not k.startswith("_")}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = dict(model=args.model, tag=args.tag, n_units=len(units), n_scenarios=len(scs), calls=eng.calls,
                in_tokens=eng.in_tok, out_tokens=eng.out_tok, seconds=round(time.time() - t0, 1), seed=args.seed,
                data=args.data)
    (out / f"{args.tag}.meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
