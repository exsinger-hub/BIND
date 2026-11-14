"""Validate binding-drift pilot scenarios against GEN_SPEC.md.

usage: python validate.py <dir_or_files...>
Prints one line per problem; exit code 1 if any file fails.
"""
import json
import sys
from pathlib import Path

KEYS = ["scenario_id", "author", "domain", "expr_category", "referent_type", "cue", "writers",
        "memory_B", "context", "instruction", "expression", "slot", "gold_op", "named_instruction",
        "concurrent_relevant", "concurrent_unrelated", "resolution", "rationale"]
CATS = {"role", "ordinal", "attribute", "anaphora", "temporal", "other"}


def check_change(name, ch, ids, errs):
    for k in ("by", "utterance", "patch", "new_records"):
        if k not in ch:
            errs.append(f"{name}: missing {k}")
            return set()
    new_ids = set()
    for r in ch["new_records"]:
        if not {"id", "type", "fields"} <= set(r):
            errs.append(f"{name}: bad new record {r}")
            continue
        new_ids.add(r["id"])
    for p in ch["patch"]:
        if not {"id", "field", "value"} <= set(p):
            errs.append(f"{name}: bad patch {p}")
        elif p["id"] not in ids | new_ids:
            errs.append(f"{name}: patch on unknown id {p['id']}")
        elif not isinstance(p["value"], str):
            errs.append(f"{name}: non-string patch value")
    if not ch["patch"] and not ch["new_records"]:
        errs.append(f"{name}: empty change")
    return new_ids


def validate(d):
    errs = []
    missing = [k for k in KEYS if k not in d]
    if missing:
        return [f"missing keys {missing}"]
    if d["expr_category"] not in CATS:
        errs.append("bad expr_category")
    if d["referent_type"] not in ("fixed", "dynamic"):
        errs.append("bad referent_type")
    if d["cue"] not in ("explicit", "implicit", "none"):
        errs.append("bad cue")
    if d["slot"] not in ("target", "value"):
        errs.append("bad slot")
    mem = d["memory_B"]
    if not 4 <= len(mem) <= 12:
        errs.append(f"memory_B has {len(mem)} records")
    ids = set()
    for r in mem:
        if not {"id", "type", "fields"} <= set(r):
            errs.append(f"bad record {r}")
            continue
        if r["id"] in ids:
            errs.append(f"dup id {r['id']}")
        ids.add(r["id"])
        if not all(isinstance(v, str) for v in r["fields"].values()):
            errs.append(f"non-string field in {r['id']}")
    if d["expression"] not in d["instruction"]:
        errs.append("expression not a substring of instruction")
    if d["expression"] in d["named_instruction"] and d["expression"].lower() not in ("it",):
        errs.append("named_instruction still contains expression")
    new_rel = check_change("concurrent_relevant", d["concurrent_relevant"], ids, errs)
    check_change("concurrent_unrelated", d["concurrent_unrelated"], ids, errs)
    res = d["resolution"]
    if not {"in_B", "in_C", "intended"} <= set(res):
        errs.append("resolution missing keys")
        return errs
    if res["in_B"] not in ids:
        errs.append("resolution.in_B not in memory_B")
    if res["in_C"] not in ids | new_rel:
        errs.append("resolution.in_C unknown")
    if res["in_B"] == res["in_C"]:
        errs.append("in_B == in_C (relevant change does not change denotation)")
    want = res["in_B"] if d["referent_type"] == "fixed" else res["in_C"]
    if res["intended"] != want:
        errs.append("resolution.intended inconsistent with referent_type")
    g = d["gold_op"]
    if not {"target", "field", "value"} <= set(g):
        errs.append("gold_op missing keys")
    else:
        slot_val = g["target"] if d["slot"] == "target" else g["value"]
        if slot_val != res["intended"]:
            errs.append(f"gold_op.{d['slot']} != resolution.intended")
        if g["target"] not in ids | new_rel:
            errs.append("gold_op.target unknown id")
    for t in d["context"]:
        if t.get("speaker") not in ("user", "assistant") or not isinstance(t.get("text"), str):
            errs.append("bad context turn")
    if len(d["context"]) > 4:
        errs.append("context longer than 4 turns")
    return errs


def main(paths):
    files = []
    for p in map(Path, paths):
        files += sorted(p.glob("*.json")) if p.is_dir() else [p]
    bad = 0
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            errs = validate(d)
        except Exception as e:  # noqa: BLE001
            errs = [f"unreadable: {e}"]
        for e in errs:
            print(f"{f.name}: {e}")
        bad += bool(errs)
    print(f"{len(files) - bad}/{len(files)} valid")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
