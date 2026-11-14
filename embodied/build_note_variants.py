"""E2 note ablations: derive instruction-file variants of the memory-grounded split (same utterances, notes changed).

The relevant note is recovered by construction: fixed -> the note that contains the read-time id ref_B;
dynamic -> the note without any object id (the distractor id-note always carries one).
Variants: nodistract (relevant note only; ambig gets no note), conflict (fixed and dynamic both get the ref AND att
note -> gold = ask), nonotes (no memory at all), clutter (two extra distractors borrowed from other episodes),
reversed (note order reversed).
usage: python build_note_variants.py --instr instructions_grounded.jsonl --episodes episodes_v1 --out instr_var/
"""
import argparse
import json
import random
import re
from pathlib import Path

ID = re.compile(r"\bobj\d+\b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instr", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    refB = {}
    for p in Path(a.episodes).glob("*/episode.json"):
        e = json.loads(p.read_text())
        refB[e["episode_id"]] = e["ref_B"]
    rows = [json.loads(l) for l in open(a.instr, encoding="utf-8")]
    rel = {}
    for r in rows:
        f = [n for n in r["fixed"]["notes"] if re.search(rf"\b{refB[r['episode_id']]}\b", n)]
        d = [n for n in r["dynamic"]["notes"] if not ID.search(n)]
        assert len(f) == 1 and len(d) == 1, r["episode_id"]
        rel[r["episode_id"]] = (f[0], d[0])
    pool = [n for r in rows for n in r["ambig"]["notes"]]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def write(name, fn):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                rng = random.Random(f"{name}|{r['episode_id']}")
                r2 = json.loads(json.dumps(r))
                for k in ("fixed", "dynamic", "ambig"):
                    r2[k]["notes"] = fn(r, k, rng)
                f.write(json.dumps(r2, ensure_ascii=False) + "\n")

    def nodistract(r, k, rng):
        fr, dr = rel[r["episode_id"]]
        return {"fixed": [fr], "dynamic": [dr], "ambig": []}[k]

    def conflict(r, k, rng):
        if k == "ambig":
            return r[k]["notes"]
        return rng.sample(list(rel[r["episode_id"]]), 2)

    def clutter(r, k, rng):
        extra = [n for n in rng.sample(pool, 6) if n not in r[k]["notes"]][:2]
        return rng.sample(r[k]["notes"] + extra, len(r[k]["notes"]) + len(extra))

    write("nodistract", nodistract)
    write("conflict", conflict)
    write("nonotes", lambda r, k, rng: [])
    write("clutter", clutter)
    write("reversed", lambda r, k, rng: r[k]["notes"][::-1])
    print("wrote", sorted(p.name for p in out.glob("*.jsonl")))


if __name__ == "__main__":
    main()
