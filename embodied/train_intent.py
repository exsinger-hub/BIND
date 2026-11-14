"""Distilled referential/attributive intent classifier (cross-encoder, e.g. DeBERTa-v3).
Data jsonl rows: {"key": ..., "desc": "...", "conv": "...", "label": 1 (referential) | 0 (attributive)}
usage: python train_intent.py --model DIR --train train.jsonl --tests impl.jsonl a1.jsonl ... --out intent_out
Writes <out>/pred_<testname>.jsonl (key, label, p_ref) and prints accuracy per test set.
"""
import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


def load(fn):
    return [json.loads(l) for l in open(fn, encoding="utf-8")]


def batches(tok, rows, bs, shuffle, max_len):
    idx = list(range(len(rows)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        b = [rows[j] for j in idx[i:i + bs]]
        enc = tok([r["desc"] for r in b], [r["conv"] for r in b], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt")
        yield b, enc, torch.tensor([int(r["label"]) for r in b])


@torch.no_grad()
def predict(model, tok, rows, max_len, dev):
    model.eval()
    out = []
    for b, enc, y in batches(tok, rows, 64, False, max_len):
        logits = model(**{k: v.to(dev) for k, v in enc.items()}).logits.float()
        p = torch.softmax(logits, -1)[:, 1].cpu().tolist()
        out += [dict(key=r["key"], label=int(r["label"]), p_ref=pp) for r, pp in zip(b, p)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--tests", nargs="+", required=True)
    ap.add_argument("--out", default="intent_out")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=2).to(dev)
    tr = load(a.train)
    random.shuffle(tr)
    nv = max(50, len(tr) // 10)
    val, tr = tr[:nv], tr[nv:]
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * ((len(tr) + a.bs - 1) // a.bs)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    for ep in range(a.epochs):
        model.train()
        tot = 0.0
        for b, enc, y in batches(tok, tr, a.bs, True, a.max_len):
            loss = model(**{k: v.to(dev) for k, v in enc.items()}, labels=y.to(dev)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            opt.zero_grad()
            tot += loss.item()
        pv = predict(model, tok, val, a.max_len, dev)
        acc = sum((p["p_ref"] > .5) == (p["label"] == 1) for p in pv) / len(pv)
        print(json.dumps(dict(epoch=ep, train_loss=round(tot, 3), val_acc=round(acc, 4), n_val=len(pv))), flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for fn in a.tests:
        rows = load(fn)
        pr = predict(model, tok, rows, a.max_len, dev)
        name = Path(fn).stem
        with open(out / f"pred_{name}.jsonl", "w") as f:
            for p in pr:
                f.write(json.dumps(p) + "\n")
        acc = sum((p["p_ref"] > .5) == (p["label"] == 1) for p in pr) / max(1, len(pr))
        accf = [(p["p_ref"] > .5) for p in pr if p["label"] == 1]
        accd = [(p["p_ref"] <= .5) for p in pr if p["label"] == 0]
        print(json.dumps(dict(test=name, n=len(pr), acc=round(acc, 4),
                              acc_ref=round(sum(accf) / max(1, len(accf)), 4),
                              acc_attr=round(sum(accd) / max(1, len(accd)), 4))), flush=True)


if __name__ == "__main__":
    main()
