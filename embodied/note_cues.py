"""Q1/Q3: how far do surface cues of the memory-grounded notes go? Rule classifiers over the notes only.
R1 any-id      : any note contains an object id -> referential, else attributive
R2 draft-id    : a note contains the read-time draft id (from the grounding VLM) -> referential, else attributive
R3 gold-id     : same with the gold read-time referent (upper bound of R2)
R4 noun+id     : a note mentions the target noun AND an object id -> referential; target noun without id -> attributive;
                 otherwise referential (wording default)
usage: python note_cues.py --instr instructions_grounded.jsonl --episodes episodes_v1 --raw res.jsonl [--notes-file alt]
"""
import argparse
import json
import re
from collections import Counter

ID = re.compile(r"\bobj\d+\b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instr", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--raw", required=True, help="grounded raw file of one VLM (for the draft id)")
    a = ap.parse_args()
    import importlib.util
    spec = importlib.util.spec_from_file_location("rv", "run_vlm.py")
    rv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rv)
    draft = {}
    for l in open(a.raw):
        r = json.loads(l)
        if r.get("draft_raw") is not None:
            op = rv.get_op(rv.parse_json(r["draft_raw"]))
            draft[(r["episode_id"], r["intent"])] = (op or {}).get("target")
    rows = [json.loads(l) for l in open(a.instr, encoding="utf-8")]
    res = {k: Counter() for k in ("R1", "R2", "R3", "R4")}
    for r in rows:
        ep = json.load(open(f"{a.episodes}/{r['episode_id']}/episode.json"))
        noun = re.sub(r"(?<!^)(?=[A-Z])", " ", ep["target_type"]).lower()
        for intent in ("fixed", "dynamic", "ambig"):
            notes = r[intent]["notes"]
            d = draft.get((r["episode_id"], intent))
            preds = {
                "R1": "fixed" if any(ID.search(n) for n in notes) else "dynamic",
                "R2": "fixed" if d and any(re.search(rf"\b{d}\b", n) for n in notes) else "dynamic",
                "R3": "fixed" if any(re.search(rf"\b{ep['ref_B']}\b", n) for n in notes) else "dynamic",
            }
            rel_id = any(noun in n.lower() and ID.search(n) for n in notes)
            rel_noid = any(noun in n.lower() and not ID.search(n) for n in notes)
            preds["R4"] = "fixed" if rel_id else "dynamic" if rel_noid else "fixed"
            for k, p in preds.items():
                res[k][(intent, p)] += 1
    for k, c in res.items():
        ref = c[("fixed", "fixed")] / 566
        att = c[("dynamic", "dynamic")] / 566
        amb_ref = c[("ambig", "fixed")] / 566
        print(f"{k}: REF {100*ref:.1f} ATT {100*att:.1f} mean {50*(ref+att):.1f} | AMB->referential {100*amb_ref:.1f}")


if __name__ == "__main__":
    main()
