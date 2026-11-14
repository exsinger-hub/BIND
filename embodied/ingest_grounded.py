"""Validate memory-grounded items and write instructions_grounded.jsonl.

Each episode gets three variants with the SAME utterance:
  fixed   : notes = [ref_note(<OBJ>=ref_B), distractor rule about another object type]
  dynamic : notes = [att_note,             distractor id-note about another visible object]
  ambig   : notes = [distractor rule, distractor id-note]        (no relevant note -> genuinely ambiguous)
Note order is shuffled per episode. Gold intent is set by construction.
usage: python ingest_grounded.py instr/grounded/out_*.json --episodes eps_json --dest instr/grounded/instructions_grounded.jsonl
"""
import argparse
import json
import random
import re
from pathlib import Path

BANNED = re.compile(r"\b(whichever|whatever|whoever|at that point|when you get (there|to it)|at the time|by then|any|"
                    r"currently|then|that one|this one|every|each|always|rule|routine)\b", re.I)
NAMES = ["Mia", "Sam", "Leo", "Ana", "Kai", "Noor", "Ivy", "Raj", "Zoe", "Tom"]
TIMES = ["07:40", "08:15", "09:05", "10:30", "12:10", "13:45", "15:20", "16:55", "18:05", "19:30"]
RULE_PLACES = ["the counter", "the shelf", "the table", "the sink", "the cabinet", "the drawer"]
ID_REMARKS = ["has a loose part, check it later", "is Grandma's, handle with care", "was borrowed from the neighbours",
              "needs a wipe before guests arrive", "was just cleaned", "should not go in the dishwasher"]


def nice(t):
    return re.sub(r"(?<!^)(?=[A-Z])", " ", t).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--dest", required=True)
    a = ap.parse_args()
    eps = {}
    for p in Path(a.episodes).glob("*.json"):
        e = json.loads(p.read_text())
        eps[e["episode_id"]] = e
    kept, bad = [], 0
    for fn in a.files:
        t = Path(fn).read_text(encoding="utf-8")
        arr = json.loads(t[t.find("["): t.rfind("]") + 1])
        for r in arr:
            e = eps.get(r.get("episode_id"))
            errs = []
            if e is None:
                errs.append("unknown episode")
            else:
                ph, tn = e["phrase"].lower(), e["target_name"].lower()
                rel = ph.replace("the ", "", 1).replace(tn, "", 1).strip()
                u, rn, an = r.get("utterance", ""), r.get("ref_note", ""), r.get("att_note", "")
                if ph not in u.lower():
                    errs.append("phrase not in utterance")
                m = BANNED.search(u)
                if m:
                    errs.append(f"utterance has banned '{m.group(0)}'")
                if rn.count("<OBJ>") != 1:
                    errs.append("ref_note needs exactly one <OBJ>")
                if ph in rn.lower():
                    errs.append("ref_note contains phrase")
                if "<OBJ>" in an or re.search(r"\bobj\d+", an):
                    errs.append("att_note has an id")
                if not (tn in an.lower() and rel in an.lower()):
                    errs.append("att_note lacks phrase parts")
            if errs:
                bad += 1
                print(r.get("episode_id"), errs)
                continue
            rng = random.Random(e["episode_id"])
            cand = set(e["candidates"])
            others = [x for x in e["records_B"] if x["id"] not in cand and x["type"] != e["target_type"]]
            other_types = sorted({x["type"] for x in e["records_B"] if x["type"] != e["target_type"]}) or ["Box"]
            o_id = rng.choice(others)["id"] if others else "obj99"
            d_rule = (f"House rule (set by {rng.choice(NAMES)}): during cleanup, the {nice(rng.choice(other_types))} "
                      f"on {rng.choice(RULE_PLACES)} goes to {rng.choice(RULE_PLACES)}.")
            d_id = f"[{rng.choice(TIMES)}] {rng.choice(NAMES)}: {o_id} {rng.choice(ID_REMARKS)}."
            ref_note = r["ref_note"].replace("<OBJ>", e["ref_B"])
            v = lambda notes: {"context": [], "instruction": r["utterance"], "notes": rng.sample(notes, len(notes))}
            kept.append(dict(episode_id=e["episode_id"], action=r.get("action", ""), destination=r.get("destination", ""),
                             fixed=v([ref_note, d_rule]), dynamic=v([r["att_note"], d_id]), ambig=v([d_rule, d_id])))
    with open(a.dest, "w", encoding="utf-8", newline="\n") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"kept {len(kept)}, rejected {bad}")


if __name__ == "__main__":
    main()
