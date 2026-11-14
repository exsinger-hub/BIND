"""Dedicated binding-mode (referential vs attributive) classification, text-only, vLLM.
usage: python mode_classify.py --model M --tag T --instr instructions.jsonl [--cot] --out modes/
"""
import argparse
import json
import re
from pathlib import Path

PROMPT = """A user is talking to a household robot. Their request picks out an object with a description: "{phrase}".

Decide how the user means that description:
- REFERENTIAL: the user means the specific object the description picks out right now, in what they are looking at. If that object later moves or changes so the description no longer fits it, the user still means that same object. Typical cues: something happened to that particular object, it is theirs, they are pointing at or looking at it, they say "that one".
- ATTRIBUTIVE: the user means whatever object fits the description at the moment the robot acts. If things change, the robot should act on whichever object fits then. Typical cues: "whichever", "whatever", "when you get there", "at that point", standing rules.

Conversation (last line is the request):
{conv}

{tail}"""
DIRECT = 'Answer with JSON only: {"mode": "referential" or "attributive"}'
COT = ('First think step by step in at most 3 sentences about the cues, then give the answer.\n'
       'End with a line of JSON: {"mode": "referential" or "attributive"}')


STRUCT = """A user is talking to a household robot. Several robots share one memory that stores notes. The user's
request picks out an object with the description: "{phrase}".

Decide whether the user means
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
   mean ATTRIBUTIVE, otherwise REFERENTIAL.{ask_rule}

{conv}

Work through the steps briefly, then end with one line of JSON: {{"mode": {modes}}}"""
# E2 prompt ablations of the structured procedure
NOFILTER = ("1. List the memory notes.",
            STRUCT[STRUCT.index("1. List the memory notes"):STRUCT.index("completely.") + len("completely.")])
NOIDRULE_OLD = STRUCT[STRUCT.index("2. If a related note"):STRUCT.index("4. If no note")]
NOIDRULE_NEW = ("2. Use the related notes to decide whether the user means one specific object or whatever fits the\n"
                "   description.\n")
ASK_RULE = ("\n5. Exception to step 4: if the request refers back to something said or noted earlier (\"like I mentioned\","
            " \"as noted\", \"per our conversation\") but no note is related, the answer is UNCLEAR - the robot must ask.")


# v9-review (nov #1) prompt-sensitivity variants: no procedure, no event/rule heuristics; answer A (= referential) / B
NEUTRAL = """A user asks a household robot to act on one object, picked out by a description. Several robots share one
memory that also stores notes written earlier by household members. Between the request and the moment the robot acts,
another robot may move or change things, so the description may come to fit a different object.

Using the request and the stored notes, decide what the user means:
(A) the particular object the description picked out when the user spoke - the robot should act on that same object
    even if it has since moved or changed;
(B) whichever object fits the description at the moment the robot acts.

Description in the request: "{phrase}"
{conv}

Think briefly, then end with one line of JSON: {{"answer": "A" or "B"}}"""
COUNTERFACTUAL = """A user asks a household robot to act on one object, picked out by the description "{phrase}". Several
robots share one memory that also stores notes written earlier by household members.

{conv}

Suppose that, before this robot acts, another robot moves or changes things so that the object the description picked
out when the user spoke no longer fits it, and a different object now fits the description. Given the request and the
notes, on which object should the robot act?
(A) the object the description picked out when the user spoke;
(B) the object that fits the description when the robot acts.

Think briefly, then end with one line of JSON: {{"answer": "A" or "B"}}"""


def swap_ab(tpl):
    """v10 review (exp #9 / nov #9): same semantic options, opposite letters and order (A = description now, B = original)."""
    i, j = tpl.index("(A) "), tpl.index("(B) ")
    k = tpl.index("\n\n", j)
    a_txt, b_txt = tpl[i + 4:j].rstrip().rstrip(";"), tpl[j + 4:k].rstrip(".")
    return tpl[:i] + f"(A) {b_txt};\n(B) {a_txt}." + tpl[k:]


def conv(v):
    c = "\n".join([f'{t["speaker"]}: {t["text"]}' for t in v.get("context", [])] + [f'user: {v["instruction"]}'])
    if v.get("notes"):
        c = "Notes stored in the robots' shared memory:\n" + "\n".join(f"- {n}" for n in v["notes"]) + "\n\n" + c
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--instr", required=True)
    ap.add_argument("--phrases", required=True, help="eps_json dir or episodes dir with episode.json")
    ap.add_argument("--cot", action="store_true")
    ap.add_argument("--out", default="modes")
    ap.add_argument("--think", action="store_true", help="enable model thinking (Qwen3)")
    ap.add_argument("--structured", action="store_true", help="memory-note matching procedure prompt")
    ap.add_argument("--ask", action="store_true", help="structured prompt may answer unclear (ask the user)")
    ap.add_argument("--gpu-util", type=float, default=0.9)
    ap.add_argument("--keys", help="json list of episode|intent keys to restrict to")
    ap.add_argument("--variant", default="full", choices=["full", "nofilter", "noidrule", "noteid", "neutral", "counterfactual"], help="E2 prompt ablation; noteid = full procedure that also returns the object id the related note binds (v7 review exp #5)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=1, help="samples per item (E8 self-consistency)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--vllm-kw", default=None, help="JSON dict of extra vllm.LLM kwargs")
    ap.add_argument("--suffix", default="", help="extra tag appended to the output file name")
    ap.add_argument("--swap-ab", action="store_true", help="neutral/counterfactual: swap option letters/order (A = ATT, B = REF)")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    rows = [json.loads(l) for l in open(a.instr, encoding="utf-8")]
    phr = {}
    for p in Path(a.phrases).glob("*/episode.json"):
        e = json.loads(p.read_text())
        phr[e["episode_id"]] = e["phrase"]
    items, prompts = [], []
    keys = set(json.load(open(a.keys))) if a.keys else None
    for r in rows:
        if r["episode_id"] not in phr:
            continue
        for intent in [i for i in ("fixed", "dynamic", "ambig") if i in r]:
            if keys is not None and f'{r["episode_id"]}|{intent}' not in keys:
                continue
            items.append((r["episode_id"], intent))
            if a.variant in ("neutral", "counterfactual"):
                tpl = NEUTRAL if a.variant == "neutral" else COUNTERFACTUAL
                prompts.append((swap_ab(tpl) if a.swap_ab else tpl).format(phrase=phr[r["episode_id"]], conv=conv(r[intent])))
            elif a.structured:
                tpl = STRUCT
                if a.variant == "nofilter":
                    tpl = tpl.replace(NOFILTER[1], NOFILTER[0])
                elif a.variant == "noidrule":
                    tpl = tpl.replace(NOIDRULE_OLD, NOIDRULE_NEW).replace("4. If no note", "3. If no note")
                elif a.variant == "noteid":
                    tpl = tpl.replace('end with one line of JSON: {{"mode": {modes}}}',
                                      'end with one line of JSON: {{"mode": {modes}, "binding_id": "<if REFERENTIAL and '
                                      'the related note names the particular object by an id, that id; otherwise none>"}}')
                assert a.variant == "full" or tpl != STRUCT
                prompts.append(tpl.format(phrase=phr[r["episode_id"]], conv=conv(r[intent]),
                                             ask_rule=ASK_RULE if a.ask else "",
                                             modes='"referential", "attributive" or "unclear"' if a.ask else '"referential" or "attributive"'))
            else:
                prompts.append(PROMPT.format(phrase=phr[r["episode_id"]], conv=conv(r[intent]), tail=COT if a.cot else DIRECT))
    kw = dict(model=a.model, max_model_len=a.max_len, gpu_memory_utilization=a.gpu_util, trust_remote_code=True,
              seed=a.seed, enforce_eager=True, limit_mm_per_prompt={"image": 0, "video": 0})
    if a.vllm_kw:
        kw.update(json.loads(a.vllm_kw))
    llm = LLM(**kw)
    sp = SamplingParams(temperature=a.temperature, n=a.n, seed=a.seed,
                        max_tokens=4096 if a.think else (600 if (a.cot or a.structured or a.variant in ("neutral", "counterfactual")) else 30))
    ck = {"chat_template_kwargs": {"enable_thinking": bool(a.think)}} if "qwen" in a.model.lower() else {}
    outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], sp, **ck)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    ok = n = 0
    suf = ("_" + a.variant + ("_swap" if a.swap_ab else "") if a.variant in ("neutral", "counterfactual") else "") + ("_cot" if a.cot else "") + ("_struct" if a.structured else "") + ("_ask" if a.ask else "") + ("_think" if a.think else "") + a.suffix

    def pmode(txt):
        txt = re.split(r"</think>", txt)[-1]
        ab = re.findall(r'"answer"\s*:\s*"([AB])"', txt)
        if a.variant in ("neutral", "counterfactual"):
            amap = {"A": "dynamic", "B": "fixed"} if a.swap_ab else {"A": "fixed", "B": "dynamic"}
            return amap.get(ab[-1] if ab else "", "?"), txt
        m = re.findall(r'"mode"\s*:\s*"(referential|attributive|unclear)"', txt)
        mode = m[-1] if m else ("referential" if "referential" in txt.lower() and "attributive" not in txt.lower() else "attributive" if "attributive" in txt.lower() else "?")
        return {"referential": "fixed", "attributive": "dynamic", "unclear": "ask"}.get(mode, "?"), txt

    with open(Path(a.out) / f"{a.tag}{suf}.jsonl", "w") as f:
        for (eid, intent), o in zip(items, outs):
            preds = [pmode(c.text) for c in o.outputs]
            votes = [p for p, _ in preds]
            pred = max(set(votes), key=lambda v: (votes.count(v), v == votes[0]))
            ok += pred == intent
            n += 1
            bid = re.findall(r'"binding_id"\s*:\s*"(obj\d+)"', preds[0][1])
            rec = dict(episode_id=eid, intent=intent, pred=pred, raw=preds[0][1][-300:], bid=bid[-1] if bid else None,
                       ntok=sum(len(c.token_ids) for c in o.outputs) / len(o.outputs), ptok=len(o.prompt_token_ids))
            if a.n > 1:
                rec["votes"] = votes
            f.write(json.dumps(rec) + "\n")
    print(json.dumps(dict(tag=a.tag, cot=a.cot, think=a.think, instr=a.instr, acc=round(ok / n, 4), n=n)))


if __name__ == "__main__":
    main()
