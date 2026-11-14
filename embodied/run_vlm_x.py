"""EmbodiedDrift VLM runner (transformers or vLLM backend), sharded + resumable.

Units: episode x intent (fixed / dynamic / named) x cond (rel / unr); named only with rel.
Stage 1 (on B, shared across conds): draft op + binding tag; explicitization rewrite.
Stage 2 (after robot B's change, state C): P0_naive, P0, P0_aware, P0_draft, H, E.
Deterministic: P1 (keep draft if tagged fixed else P0), FREEZE (always commit the read-time object id = ID replay).
Memory modality: --mem both (images + records, default) | img | graph.

usage: python run_vlm.py --model <path> --tag qwen3vl8b --episodes episodes_v1 --instr instr/instructions.jsonl
                         --out res --shard 0/4 [--backend hf|vllm] [--mem both]
"""
import argparse
import json
import re
import time
from pathlib import Path

SYSTEM = ("You are the memory-and-planning module of household robot A. Several robots share one object memory. "
          "You turn the user's request into one action on one object. Objects are identified by ids like obj7; in the "
          "images every tracked object carries its id number as a yellow label. Answer with JSON only.")
OP_SPEC = '{"op": {"target": "<object id, e.g. obj7>", "action": "<short verb phrase>", "destination": "<where, or user>"}}'
DRAFT_SPEC = ('{"op": {"target": "<object id>", "action": "<short verb phrase>", "destination": "<where, or user>"}, '
              '"referring_expression": "<phrase in the request that picks out the object>", '
              '"resolved_id": "<object id that phrase picks out>", "kind": "fixed or dynamic"}')
AWARE_NOTE = ("Important: the user spoke while looking at version 1. For the description in the request that picks out "
              "the object (a place, a position such as 'on the left', a property such as 'dirty', a nearby object), "
              "decide what the user meant: the specific object it picked out in version 1, or whatever object it picks "
              "out when the action is carried out. Use the conversation to decide, and carry out what the user meant.\n\n")


# E7 prompt paraphrases: (retry header, translate instruction, retry tail); variant 0 = the original wording
PARA = [("Your planned action was rejected by the shared-memory version check: while you were planning, another robot "
         "changed the memory. Re-plan the user's request against the CURRENT memory.",
         "Translate the user's latest request into one action on one object.",
         "Produce the single action to carry out now."),
        ("The shared-memory version check refused your plan because another robot committed a change while you were "
         "planning. Plan the user's request again using the CURRENT memory.",
         "Turn the user's last request into a single action on a single object.",
         "Give the one action to execute now."),
        ("Commit failed: another robot modified the shared memory after you read it. Re-plan the request against the "
         "up-to-date CURRENT memory.",
         "Convert the most recent user request into one action applied to one object.",
         "State the one action you will carry out now.")]
PV = 0


def conv_text(ctx, instr):
    return "\n".join([f'{t["speaker"]}: {t["text"]}' for t in ctx] + [f"user: {instr}"])


def recs_text(recs):
    return "\n".join(json.dumps(r) for r in recs)


class Unit:
    def __init__(self, ep, ins, intent, cond, epdir):
        self.ep, self.intent, self.cond = ep, intent, cond
        v = ins[intent]
        self.ctx, self.instr = v.get("context", []), v["instruction"]
        self.notes = list(v.get("notes", []))
        self.action, self.dest = ins.get("action", ""), ins.get("destination", "")
        self.imgB = str(epdir / "B.png")
        self.imgC = str(epdir / ("C_rel.png" if cond == "rel" else "C_unr.png"))
        self.recB = ep["records_B"]
        self.recC = ep["records_C_rel"] if cond == "rel" else ep["records_C_unr"]
        self.change = ep["change_rel"] if cond == "rel" else ep["change_unr"]
        self.ref_B = ep["ref_B"]
        self.ref_C = ep["ref_C_rel"] if cond == "rel" else ep["ref_C_unr"]
        dyn = intent.startswith("dynamic")
        self.gold = self.ref_C if dyn else self.ref_B
        self.key = f'{ep["episode_id"]}|{intent}|{cond}'
        self.dkey = f'{ep["episode_id"]}|{intent}'  # stage-1 key (B-only)


# ------------------------------------------------------------------ prompt builders -> list of content parts
def mem_parts(label, img, recs, mem, notes=None):
    parts = []
    if mem in ("both", "img"):
        parts += [{"type": "text", "text": f"{label} — camera image:"}, {"type": "image", "path": img}]
    if mem in ("both", "graph"):
        parts += [{"type": "text", "text": f"{label} — object records:\n{recs_text(recs)}"}]
    if notes:
        parts += [{"type": "text", "text": f"{label} — notes stored in the shared memory:\n" +
                   "\n".join(f"- {n}" for n in notes)}]
    return parts


def p_draft(u, mem):
    return (mem_parts("SHARED MEMORY (current version)", u.imgB, u.recB, mem, u.notes) +
            [{"type": "text", "text": f"CONVERSATION WITH YOUR USER:\n{conv_text(u.ctx, u.instr)}\n\n"
              "Translate the user's latest request into one action on one object.\n"
              '"kind" is "fixed" if the phrase is meant to pick out the specific object it identifies right now, or '
              '"dynamic" if it is meant to be re-evaluated when the action is carried out.\n'
              f"Return only JSON in this format:\n{DRAFT_SPEC}"}])


def p_explicit(u, mem):
    return (mem_parts("SHARED MEMORY (current version)", u.imgB, u.recB, mem, u.notes) +
            [{"type": "text", "text": f"CONVERSATION WITH YOUR USER:\n{conv_text(u.ctx, u.instr)}\n\n"
              "Rewrite the user's latest request as a standalone instruction that another robot can execute later "
              "without this conversation. Replace descriptions that refer to a specific object with that object's id "
              "(e.g. 'obj7'). If the user means a description to be evaluated when the instruction is executed (for "
              "example 'whichever mug is by the sink at that point'), keep that description as is. Keep everything "
              'else unchanged.\nReturn only JSON in this format:\n{"rewritten": "<standalone instruction>"}'}])


MODE_PROMPT = """A user is talking to a household robot. Their latest request picks out one object with a description
(a place, a position such as 'on the far left', a property such as 'dirty', a nearby object).

Decide how the user means that description:
- REFERENTIAL: the user means the specific object the description picks out right now, in what they are looking at.
  If that object later moves or changes so the description no longer fits, the user still means that same object.
  Typical cues: something happened to that particular object, it is theirs, they are pointing at or looking at it.
- ATTRIBUTIVE: the user means whatever object fits the description at the moment the robot acts. Typical cues:
  "whichever", "whatever", "when you get there", "at that point", standing rules.

CONVERSATION (last line is the request):
{conv}

First write the description you found and think in at most 3 sentences about the cues, then end with one line of
JSON: {{"description": "...", "mode": "referential" or "attributive"}}"""


def p_mode(u, mem):
    conv = conv_text(u.ctx, u.instr)
    if u.notes:
        conv = "NOTES STORED IN THE SHARED MEMORY:\n" + "\n".join(f"- {n}" for n in u.notes) + "\n\n" + conv
    return [{"type": "text", "text": MODE_PROMPT.format(conv=conv)}]


def parse_mode(txt):
    m = re.findall(r'"mode"\s*:\s*"(referential|attributive)"', txt or "")
    return m[-1] if m else "?"


def p_naive(u, mem, instr=None, ctx=None):
    instr = u.instr if instr is None else instr
    ctx = u.ctx if ctx is None else ctx
    return (mem_parts("SHARED MEMORY (current version)", u.imgC, u.recC, mem, u.notes) +
            [{"type": "text", "text": f"CONVERSATION WITH YOUR USER:\n{conv_text(ctx, instr)}\n\n"
              f"{PARA[PV][1]}\n"
              f"Return only JSON in this format:\n{OP_SPEC}"}])


def change_text(u, ids=True):
    if not ids:
        return f'{u.change["by"]}: {u.change["log"]}'
    return f'{u.change["by"]}: {u.change["log"]} (objects touched: {", ".join(u.change["objects"])})'


def p_retry(u, mem, draft=None, aware=False, ids=True):
    parts = [{"type": "text", "text": PARA[PV][0] + "\n\n" + (AWARE_NOTE if aware else "")}]
    parts += mem_parts("MEMORY WHEN THE USER SPOKE (version 1)", u.imgB, u.recB, mem, u.notes)
    parts += [{"type": "text", "text": f"CHANGE COMMITTED SINCE THEN (version 2):\n{change_text(u, ids)}"}]
    parts += mem_parts("CURRENT MEMORY (version 2)", u.imgC, u.recC, mem, u.notes)
    t = f"CONVERSATION WITH YOUR USER:\n{conv_text(u.ctx, u.instr)}\n\n"
    if draft is not None:
        t += f"YOUR REJECTED PLAN (made against version 1):\n{json.dumps({'op': draft})}\n\n"
    t += f"{PARA[PV][2]}\nReturn only JSON in this format:\n{OP_SPEC}"
    return parts + [{"type": "text", "text": t}]


STRUCT_STEPS = """Decide whether the user means
- REFERENTIAL: one specific object, or
- ATTRIBUTIVE: whatever object fits the description at the moment the robot acts.

Follow this procedure:
1. List the memory notes and say, for each, whether it is about THIS request (same object type, same action or
   purpose) or unrelated (another object or another task). Ignore unrelated notes completely.
2. If a related note ties the request to one particular object (for example it names an object id like obj3, or
   an event that happened to one specific item), the answer is REFERENTIAL.
3. If a related note is a standing rule or policy stated in terms of the description (for example "during the
   evening reset, the mug on the coffee machine goes to the sink"), the answer is ATTRIBUTIVE.
4. If no note is related, decide from the request alone; explicit words like "whichever" or "when you get there"
   mean ATTRIBUTIVE, otherwise REFERENTIAL."""


def p_struct(u, mem):
    """Single-stage resolver given the same structured procedure as the BIND-LLM reasoner (prompt-matched control)."""
    parts = [{"type": "text", "text": PARA[PV][0] + "\n\n"}]
    parts += mem_parts("MEMORY WHEN THE USER SPOKE (version 1)", u.imgB, u.recB, mem, u.notes)
    parts += [{"type": "text", "text": f"CHANGE COMMITTED SINCE THEN (version 2):\n{change_text(u)}"}]
    parts += mem_parts("CURRENT MEMORY (version 2)", u.imgC, u.recC, mem, u.notes)
    t = (f"CONVERSATION WITH YOUR USER:\n{conv_text(u.ctx, u.instr)}\n\n"
         f'The request picks out an object with the description: "{u.ep["phrase"]}".\n' + STRUCT_STEPS +
         "\n5. If REFERENTIAL, act on the object the description picked out in version 1 (when the user spoke). If "
         "ATTRIBUTIVE, act on the object the description picks out in the current version 2.\n\n"
         f"Work through the steps briefly, then end with one line of JSON in this format:\n{OP_SPEC}")
    return parts + [{"type": "text", "text": t}]


STRUCTD_SPEC = ('{"related_notes": "<which notes are about this request, one short sentence>", '
                '"binding": "referential or attributive", '
                '"op": {"target": "<object id>", "action": "<short verb phrase>", "destination": "<where, or user>"}}')


def p_struct_draft(u, mem):
    """Draft-aware single-call control (v7 review exp #2 / nov #2): the same information BIND has at retry time (both
    snapshots, notes, change, and the stored read-time draft), neutral framing, binding decision and target in one call."""
    parts = [{"type": "text", "text": "Your planned action could not be committed: another robot changed the shared "
              "memory after you read it. Decide what to do with the request now.\n\n"}]
    parts += mem_parts("MEMORY WHEN THE USER SPOKE (version 1)", u.imgB, u.recB, mem, u.notes)
    parts += [{"type": "text", "text": f"CHANGE COMMITTED SINCE THEN (version 2):\n{change_text(u)}"}]
    parts += mem_parts("CURRENT MEMORY (version 2)", u.imgC, u.recC, mem, u.notes)
    t = (f"CONVERSATION WITH YOUR USER:\n{conv_text(u.ctx, u.instr)}\n\n"
         f"YOUR STORED PLAN (made against version 1):\n{json.dumps({'op': u.draft})}\n\n"
         f'The request picks out an object with the description: "{u.ep["phrase"]}".\n' + STRUCT_STEPS +
         "\n5. If REFERENTIAL, keep the target of your stored plan. If ATTRIBUTIVE, act on the object the description "
         "picks out in the current version 2.\n\n"
         f"Return only JSON in this format:\n{STRUCTD_SPEC}")
    return parts + [{"type": "text", "text": t}]


def p_guard(u, mem):
    """Identity-aware guarded retry (v7 review nov #5; an adaptation of post-grounding verification, not a reproduction
    of RefGuard): keep the stored target if it still satisfies the description in the current memory, else re-resolve."""
    parts = [{"type": "text", "text": "Your planned action could not be committed: another robot changed the shared "
              "memory after you read it.\n\n"}]
    parts += mem_parts("MEMORY WHEN THE USER SPOKE (version 1)", u.imgB, u.recB, mem, u.notes)
    parts += [{"type": "text", "text": f"CHANGE COMMITTED SINCE THEN (version 2):\n{change_text(u)}"}]
    parts += mem_parts("CURRENT MEMORY (version 2)", u.imgC, u.recC, mem, u.notes)
    t = (f"CONVERSATION WITH YOUR USER:\n{conv_text(u.ctx, u.instr)}\n\n"
         f"YOUR STORED PLAN (made against version 1):\n{json.dumps({'op': u.draft})}\n\n"
         f'Verify the stored target against the current version 2: if the object of your stored plan still fits the '
         f'description "{u.ep["phrase"]}" now, keep it; otherwise act on the object that fits the description now.\n'
         f"Return only JSON in this format:\n{OP_SPEC}")
    return parts + [{"type": "text", "text": t}]


def p_pred(u, mem):
    """Commitment-conditioned resolver (v10 review exp #5 / nov #6): the ATT branch is told the selected commitment
    PREDICATE(description) and only selects the current match; it does not re-decide identity vs. description."""
    return (mem_parts("SHARED MEMORY (current version)", u.imgC, u.recC, mem, u.notes) +
            [{"type": "text", "text": f"CONVERSATION WITH YOUR USER:\n{conv_text(u.ctx, u.instr)}\n\n"
              f'It has already been decided that the description "{u.ep["phrase"]}" in the request must be evaluated '
              "against the CURRENT memory: act on the object that fits this description now, even if a different object "
              "fitted it earlier. Do not try to keep an earlier object.\n"
              f"{PARA[PV][1]}\n"
              f"Return only JSON in this format:\n{OP_SPEC}"}])


def p_retry_nob(u, mem):
    """Retry prompt identical to P0 except that the version-1 camera image is withheld (records and notes kept)."""
    parts = [{"type": "text", "text": PARA[PV][0] + "\n\n"}]
    parts += mem_parts("MEMORY WHEN THE USER SPOKE (version 1)", u.imgB, u.recB, "graph" if mem == "both" else mem,
                       u.notes)
    parts += [{"type": "text", "text": f"CHANGE COMMITTED SINCE THEN (version 2):\n{change_text(u)}"}]
    parts += mem_parts("CURRENT MEMORY (version 2)", u.imgC, u.recC, mem, u.notes)
    t = f"CONVERSATION WITH YOUR USER:\n{conv_text(u.ctx, u.instr)}\n\n"
    t += f"{PARA[PV][2]}\nReturn only JSON in this format:\n{OP_SPEC}"
    return parts + [{"type": "text", "text": t}]


def p_history(u, mem):
    parts = [{"type": "text", "text": "EVENT LOG OF THE SHARED MEMORY\n"}]
    parts += mem_parts("[v1] memory state", u.imgB, u.recB, mem, u.notes)
    parts += [{"type": "text", "text": f"[event A] another robot (had read v1) committed v2:\n{change_text(u)}"}]
    parts += mem_parts("[v2] memory state", u.imgC, u.recC, mem, u.notes)
    parts += [{"type": "text", "text": "[event B] robot A (had read v1) received this conversation from its user and "
               f"must now act, after event A:\n{conv_text(u.ctx, u.instr)}\n\nDetermine the single action that correctly "
               f"carries out the request in event B.\nReturn only JSON in this format:\n{OP_SPEC}"}]
    return parts


# ------------------------------------------------------------------ parsing
def _json_objects(text):
    """All top-level balanced {...} substrings that parse as JSON objects, in order."""
    objs, depth, st = [], 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                st = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and st is not None:
                try:
                    o = json.loads(text[st: i + 1])
                    if isinstance(o, dict):
                        objs.append(o)
                except Exception:  # noqa: BLE001
                    pass
    return objs


def parse_json(text, want=("op", "target", "rewritten", "mode")):
    """Last JSON object that contains one of the expected answer keys (falls back to the last object)."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    objs = _json_objects(text)
    for o in reversed(objs):
        if any(k in o for k in want):
            return o
    return objs[-1] if objs else None


def get_op(obj):
    """Canonical op dict with target 'obj<int>' or None if the answer has no usable target."""
    if not isinstance(obj, dict):
        return None
    op = obj.get("op", obj)
    if not isinstance(op, dict) or "target" not in op:
        return None
    m = re.fullmatch(r"\s*(?:obj(?:ect)?\s*#?\s*)?(\d+)\s*", str(op.get("target", "")), flags=re.I)
    if not m:
        return None
    return {"target": f"obj{int(m.group(1))}", "action": str(op.get("action", "")),
            "destination": str(op.get("destination", ""))}


# ------------------------------------------------------------------ backends
class HFBackend:
    def __init__(self, model, max_new, bs):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(model, trust_remote_code=True)
        try:
            self.proc.tokenizer.padding_side = "left"
        except Exception:  # noqa: BLE001
            pass
        self.model = AutoModelForImageTextToText.from_pretrained(model, torch_dtype=torch.bfloat16,
                                                                 device_map="cuda:0", trust_remote_code=True).eval()
        self.max_new, self.bs = max_new, bs
        self.calls = self.in_tok = self.out_tok = 0

    def _msg(self, parts):
        content = []
        for p in parts:
            if p["type"] == "image":
                content.append({"type": "image", "url": p["path"]})
            else:
                content.append({"type": "text", "text": p["text"]})
        return [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]}, {"role": "user", "content": content}]

    def run(self, prompts):
        outs = []
        for i in range(0, len(prompts), self.bs):
            msgs = [self._msg(p) for p in prompts[i:i + self.bs]]
            kw = dict(add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", padding=True)
            try:
                inp = self.proc.apply_chat_template(msgs, enable_thinking=False, **kw)
            except TypeError:
                inp = self.proc.apply_chat_template(msgs, **kw)
            inp = inp.to(self.model.device)
            with self.torch.inference_mode():
                gen = self.model.generate(**inp, max_new_tokens=self.max_new, do_sample=False)
            new = gen[:, inp["input_ids"].shape[1]:]
            txt = self.proc.batch_decode(new, skip_special_tokens=True)
            self.calls += len(txt)
            self.in_tok += int(inp["attention_mask"].sum())
            self.out_tok += int((new != self.proc.tokenizer.pad_token_id).sum())
            outs += txt
            print(f"  {i + len(txt)}/{len(prompts)}", flush=True)
        return outs


class VLLMBackend:
    def __init__(self, model, max_new, bs, tp=1, quant=None, gpu_util=0.88, max_len=16384, extra=None,
                 temperature=0.0, seed=2026):
        from vllm import LLM, SamplingParams
        kw = dict(model=model, max_model_len=max_len, gpu_memory_utilization=gpu_util, trust_remote_code=True,
                  seed=seed, limit_mm_per_prompt={"image": 2, "video": 0}, allowed_local_media_path="/",
                  enforce_eager=True, tensor_parallel_size=tp)
        if "qwen" in model.lower():
            kw["mm_processor_kwargs"] = {"max_pixels": 1_000_000}
        if "qwen3-vl" in model.lower():
            # deepstack embeds break when an image is split across prefill chunks (vLLM masked-scatter error)
            kw["enable_chunked_prefill"] = False
            kw["max_num_batched_tokens"] = max_len
        elif "internvl" in model.lower():
            kw["mm_processor_kwargs"] = {"max_patches": 4}
        if quant:
            kw["quantization"] = quant
        if extra:
            kw.update(extra)
        self.llm = LLM(**kw)
        self.sp = SamplingParams(temperature=temperature, max_tokens=max_new, seed=seed)
        self.model = model
        self.calls = self.in_tok = self.out_tok = 0
        self.ptok = []

    def run(self, prompts):
        convs = []
        for parts in prompts:
            content = [{"type": "image_url", "image_url": {"url": "file://" + p["path"]}} if p["type"] == "image"
                       else {"type": "text", "text": p["text"]} for p in parts]
            convs.append([{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}])
        kw = {"chat_template_kwargs": {"enable_thinking": False}} if "qwen" in self.model.lower() else {}
        outs = self.llm.chat(convs, self.sp, use_tqdm=True, **kw)
        res = []
        for o in outs:
            self.calls += 1
            self.in_tok += len(o.prompt_token_ids)
            self.out_tok += len(o.outputs[0].token_ids)
            res.append(o.outputs[0].text)
            self.ptok.append((len(o.prompt_token_ids), len(o.outputs[0].token_ids)))
        return res


class MockBackend:
    """Pipeline test only: answers with the first object id mentioned in the last text part."""

    def __init__(self, *a):
        self.calls = self.in_tok = self.out_tok = 0

    def run(self, prompts):
        out = []
        for parts in prompts:
            self.calls += 1
            m = re.findall(r"obj\d+", " ".join(p.get("text", "") for p in parts))
            t = m[0] if m else "obj1"
            out.append(json.dumps({"op": {"target": t, "action": "move", "destination": "sink"},
                                   "kind": "fixed", "rewritten": "move it"}))
        return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--instr", required=True)
    ap.add_argument("--out", default="res")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--backend", default="hf", choices=["hf", "vllm", "mock"])
    ap.add_argument("--mem", default="both", choices=["both", "img", "graph", "shuf"],
                    help="shuf = images and records, but both images come from another scene (visual-necessity control)")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--intents", default="fixed,dynamic,named")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--quant", default=None)
    ap.add_argument("--gpu-util", type=float, default=0.88)
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--vllm-kw", default=None, help="JSON dict of extra vllm.LLM kwargs")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--prompt-var", type=int, default=0, choices=[0, 1, 2], help="E7 retry-prompt paraphrase")
    ap.add_argument("--episode-list", help="text file with one episode id per line (subset)")
    ap.add_argument("--configs", default="P0_naive,P0,P0_aware,P0_draft,H,P0_noid,E", help="stage-2 configs to run")
    args = ap.parse_args()
    global PV
    PV = args.prompt_var
    si, sn = map(int, args.shard.split("/"))

    instr = {}
    for line in open(args.instr, encoding="utf-8"):
        r = json.loads(line)
        instr[r["episode_id"]] = r
    if args.episode_list:
        keep = set(open(args.episode_list).read().split())
        instr = {k: v for k, v in instr.items() if k in keep}
    units = []
    for k, eid in enumerate(sorted(instr)):
        if k % sn != si:
            continue
        epdir = Path(args.episodes) / eid
        if not (epdir / "episode.json").exists():
            continue
        ep = json.loads((epdir / "episode.json").read_text())
        plan = [(i, ("rel",) if i == "named" else ("rel", "unr")) for i in args.intents.split(",")]
        for intent, conds in plan:
            if intent not in instr[eid]:
                continue
            for cond in conds:
                units.append(Unit(ep, instr[eid], intent, cond, epdir.resolve()))
    if args.mem == "shuf":  # deterministic re-pairing with an episode of another scene
        import random
        allep = sorted(p.name for p in Path(args.episodes).iterdir() if (p / "B.png").exists())
        rng = random.Random(2027)
        for u in units:
            sc = u.ep["episode_id"].split("_s")[0]
            pool = [e for e in allep if e.split("_s")[0] != sc]
            o = Path(args.episodes).resolve() / random.Random(u.ep["episode_id"]).choice(pool)
            u.imgB = str(o / "B.png")
            u.imgC = str(o / ("C_rel.png" if u.cond == "rel" else "C_unr.png"))
        del rng
    mem_run = "both" if args.mem == "shuf" else args.mem
    if args.limit:
        units = units[: args.limit]
    out = Path(args.out) / args.mem
    out.mkdir(parents=True, exist_ok=True)
    fn = out / f"{args.tag}.shard{si}of{sn}.jsonl"
    done = set()
    if fn.exists():
        done = {json.loads(l)["key"] for l in open(fn)}
    units = [u for u in units if u.key not in done]
    print(f"{len(units)} units to run (shard {si}/{sn}, {len(done)} done)", flush=True)
    if not units:
        return
    if args.backend == "vllm":
        be = VLLMBackend(args.model, args.max_new, args.bs, tp=args.tp, quant=args.quant, gpu_util=args.gpu_util,
                         max_len=args.max_len, extra=json.loads(args.vllm_kw) if args.vllm_kw else None,
                         temperature=args.temperature, seed=args.seed)
    else:
        be = {"hf": HFBackend, "mock": MockBackend}[args.backend](args.model, args.max_new, args.bs)
    t0 = time.time()
    # stage 1: one draft + one rewrite per (episode, intent), on B
    s1 = {}
    for u in units:
        s1.setdefault(u.dkey, u)
    s1u = list(s1.values())
    n1 = len(s1u)
    r1 = be.run([p_draft(u, mem_run) for u in s1u] + [p_explicit(u, mem_run) for u in s1u] +
                [p_mode(u, mem_run) for u in s1u])
    pt1 = list(getattr(be, "ptok", []))
    tok1 = {u.dkey: {"draft": pt1[i], "explicit": pt1[n1 + i], "mode": pt1[2 * n1 + i]}
            for i, u in enumerate(s1u)} if len(pt1) == 3 * n1 else {}
    dr = {u.dkey: r1[i] for i, u in enumerate(s1u)}
    er = {u.dkey: r1[n1 + i] for i, u in enumerate(s1u)}
    mr = {u.dkey: r1[2 * n1 + i] for i, u in enumerate(s1u)}
    # stage 2
    jobs = []
    for i, u in enumerate(units):
        dobj = parse_json(dr[u.dkey]) or {}
        u.draft = get_op(dobj)
        u.kind = str(dobj.get("kind", "")).lower().strip()
        eobj = parse_json(er[u.dkey]) or {}
        u.rewritten = eobj.get("rewritten") if isinstance(eobj, dict) else None
        rw = u.rewritten if isinstance(u.rewritten, str) and u.rewritten.strip() else u.instr
        cand = {"P0_naive": lambda: p_naive(u, mem_run), "P0": lambda: p_retry(u, mem_run),
                "P0_aware": lambda: p_retry(u, mem_run, aware=True),
                "P0_draft": lambda: p_retry(u, mem_run, draft=u.draft), "H": lambda: p_history(u, mem_run),
                "P0_noid": lambda: p_retry(u, mem_run, ids=False), "E": lambda: p_naive(u, mem_run, instr=rw, ctx=[]),
                "P0_struct": lambda: p_struct(u, mem_run), "P0_noB": lambda: p_retry_nob(u, mem_run),
                "P0_structd": lambda: p_struct_draft(u, mem_run), "P0_guard": lambda: p_guard(u, mem_run),
                "P0_pred": lambda: p_pred(u, mem_run)}
        jobs += [(i, c, cand[c]()) for c in args.configs.split(",")]
    n0 = len(getattr(be, "ptok", []))
    r2 = be.run([j[2] for j in jobs])
    pt2 = list(getattr(be, "ptok", []))[n0:]
    outs, toks = {}, {}
    for k, ((i, cfg, _), txt) in enumerate(zip(jobs, r2)):
        outs.setdefault(i, {})[cfg] = (get_op(parse_json(txt)), txt)
        if len(pt2) == len(jobs):
            toks.setdefault(i, {})[cfg] = pt2[k]
    with open(fn, "a", encoding="utf-8") as f:
        for i, u in enumerate(units):
            o = {k: v[0] for k, v in outs[i].items()}
            o["draft"] = u.draft
            o["P1"] = u.draft if (u.kind == "fixed" and u.draft) else o.get("P0")
            o["FREEZE"] = u.draft
            mode = parse_mode(mr[u.dkey])
            o["BIND"] = u.draft if (mode == "referential" and u.draft) else o.get("P0_naive")
            o["BIND_E"] = u.draft if (mode == "referential" and u.draft) else o.get("E")
            rec = dict(key=u.key, episode_id=u.ep["episode_id"], family=u.ep["family"], room=u.ep["room"],
                       intent=u.intent, cond=u.cond, ref_B=u.ref_B, ref_C=u.ref_C, gold=u.gold, kind=u.kind,
                       rewritten=u.rewritten, mode=mode, mode_raw=mr[u.dkey][-400:], out=o,
                       raw={k: v[1] for k, v in outs[i].items()},
                       draft_raw=dr[u.dkey], tok={**tok1.get(u.dkey, {}), **toks.get(i, {})})
            f.write(json.dumps(rec) + "\n")
    meta = dict(model=args.model, tag=args.tag, shard=args.shard, mem=args.mem, units=len(units), calls=be.calls,
                in_tokens=be.in_tok, out_tokens=be.out_tok, seconds=round(time.time() - t0, 1))
    (out / f"{args.tag}.shard{si}of{sn}.meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
