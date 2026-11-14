"""EmbodiedDrift episode generator (AI2-THOR, CloudRendering).

Object identity is tracked by the unique object `name` (AI2-THOR objectIds encode position and change when
SetObjectPoses respawns an object).

For each (scene, seed):
  spawn duplicates of a target type T, randomize materials, pick a viewpoint where >=2 T instances are visible,
  then for each description family (on / near / left / right / dirty / filled) whose B-resolution is unique:
    relevant change  : swap the B-referent with another candidate (or move the state to another candidate) so the
                       description resolves to a different instance in C
    unrelated change : swap two non-candidate objects; resolution unchanged
  Resolutions are RE-COMPUTED from post-change metadata, and the episode is kept only if all checks pass.
Outputs: <out>/<episode_id>/{B,C_rel,C_unr}.png (+ *_raw.png) and episode.json

usage: python gen_episodes.py --scenes FloorPlan1 FloorPlan2 ... --seeds 0 1 --out episodes
"""
import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from PIL import Image, ImageDraw, ImageFont

TARGETS = {
    "kitchen": ["Mug", "Cup", "Apple", "Tomato", "Bowl", "Plate", "Potato", "Egg", "Bottle", "Pot", "Pan",
                "Kettle", "SaltShaker", "WineBottle"],
    "living": ["Book", "RemoteControl", "Vase", "Statue", "Box", "CreditCard", "KeyChain", "Newspaper",
               "Watch", "Pillow"],
    "bedroom": ["Book", "CellPhone", "Mug", "AlarmClock", "CD", "Pencil", "Pen", "KeyChain", "Pillow",
                "TeddyBear", "BaseballBat", "BasketBall"],
    "bathroom": ["SoapBar", "SoapBottle", "ToiletPaper", "Candle", "SprayBottle", "Cloth", "TissueBox"],
}
ANCHORS = ["CoffeeMachine", "Microwave", "Toaster", "SinkBasin", "StoveBurner", "Fridge", "Television", "Sofa",
           "ArmChair", "FloorLamp", "DeskLamp", "Laptop", "Bed", "Dresser", "Desk", "Toilet", "Bathtub",
           "Faucet", "Window", "Painting", "HousePlant", "GarbageCan", "Chair", "Stool", "Shelf"]
DIRTYABLE = {"Mug", "Cup", "Bowl", "Plate", "Pot", "Pan"}
FILLABLE = {"Mug", "Cup", "Bowl", "Pot", "Kettle", "WineBottle", "Bottle"}
INSIDE = {"SinkBasin", "Fridge", "Drawer", "Cabinet", "Box", "Microwave", "GarbageCan", "Bathtub", "BathtubBasin",
          "Safe", "Pot", "Pan", "Bowl", "Cup", "Mug"}

REASONS = {}


def why(k):
    REASONS[k] = REASONS.get(k, 0) + 1


def room_type(scene):
    n = int(re.sub(r"\D", "", scene))
    return "kitchen" if n < 200 else "living" if n < 300 else "bedroom" if n < 400 else "bathroom"


def nice(t):
    t = {"SinkBasin": "Sink", "StoveBurner": "Stove", "CounterTop": "Counter", "BathtubBasin": "Bathtub",
         "TVStand": "TVStand"}.get(t, t)
    return re.sub(r"(?<!^)(?=[A-Z])", " ", t).lower()


def otype(oid_or_name):
    return re.split(r"[|_]", oid_or_name)[0]


class Gen:
    def __init__(self, width, height):
        self.c = Controller(platform=CloudRendering, scene="FloorPlan1", width=width, height=height,
                            fieldOfView=90, renderInstanceSegmentation=True, gridSize=0.25, snapToGrid=False,
                            visibilityDistance=2.5)
        try:
            self.font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
        except Exception:  # noqa: BLE001
            self.font = ImageFont.load_default()

    # ------------------------------------------------------------------ helpers (name-based identity)
    @property
    def ev(self):
        return self.c.last_event

    @staticmethod
    def byname(ev):
        return {o["name"]: o for o in ev.metadata["objects"]}

    @staticmethod
    def mask(ev, name):
        o = Gen.byname(ev).get(name)
        if o is None or not ev.instance_masks or o["objectId"] not in ev.instance_masks:
            return None
        return ev.instance_masks[o["objectId"]]

    @staticmethod
    def area(ev, name):
        m = Gen.mask(ev, name)
        return 0 if m is None else int(m.sum())

    def step_on(self, action, name, **kw):
        o = self.byname(self.ev).get(name)
        if o is None:
            return False
        return self.c.step(action=action, objectId=o["objectId"], forceAction=True, **kw).metadata["lastActionSuccess"]

    # ------------------------------------------------------------------ scene setup
    def setup(self, scene, seed, T):
        c = self.c
        c.reset(scene=scene)
        ev = c.step(action="InitialRandomSpawn", randomSeed=seed, forceVisible=True, numPlacementAttempts=25,
                    placeStationary=True, numDuplicatesOfType=[{"objectType": T, "count": 3}])
        if not ev.metadata["lastActionSuccess"]:
            return None
        c.step(action="RandomizeMaterials")
        cands = [o["name"] for o in self.ev.metadata["objects"] if o["objectType"] == T]
        return cands if len(cands) >= 2 else None

    def best_view(self, cands, rng):
        c = self.c
        pos = c.step(action="GetReachablePositions").metadata["actionReturn"] or []
        rng.shuffle(pos)
        best = None
        for p in pos[:70]:
            for rot in (0, 90, 180, 270):
                e = c.step(action="Teleport", position=p, rotation=dict(x=0, y=rot, z=0), horizon=30, standing=True)
                if not e.metadata["lastActionSuccess"]:
                    continue
                areas = sorted((self.area(e, n) for n in cands), reverse=True)
                sc = areas[1] + 0.3 * (areas[2] if len(areas) > 2 else 0)  # >=2 clearly visible, prefer 3
                if best is None or sc > best[0]:
                    best = (sc, p, rot)
        if best is None or best[0] < 500:
            return None
        return dict(position=best[1], rotation=dict(x=0, y=best[2], z=0), horizon=30, standing=True)

    def look(self, view):
        return self.c.step(action="Teleport", **view)

    # ------------------------------------------------------------------ resolution functions
    @staticmethod
    def resolve(fam, arg, cands, ev):
        """Return the candidate NAME the description denotes in ev, or None if not unique."""
        bn = Gen.byname(ev)
        cs = [bn[n] for n in cands if n in bn]
        if fam == "on":  # arg = receptacle TYPE; unique at the type level ("the plate on the counter")
            hits = [o["name"] for o in cs if any(otype(r) == arg for r in (o.get("parentReceptacles") or []))]
            return hits[0] if len(hits) == 1 else None
        if fam == "near":  # arg = anchor name
            a = bn.get(arg)
            if a is None:
                return None
            ap = np.array([a["axisAlignedBoundingBox"]["center"][k] for k in "xyz"])
            d = sorted((float(np.linalg.norm(np.array([o["position"][k] for k in "xyz"]) - ap)), o["name"]) for o in cs)
            if len(d) < 2 or d[1][0] < 1.25 * d[0][0] or d[1][0] - d[0][0] < 0.08:
                return None
            return d[0][1]
        if fam in ("left", "right"):  # image-plane order among visible candidates
            xs = []
            for o in cs:
                m = Gen.mask(ev, o["name"])
                if m is not None and m.sum() >= 120:
                    xs.append((float(np.nonzero(m)[1].mean()), o["name"]))
            if len(xs) < 2:
                return None
            xs.sort(reverse=(fam == "right"))
            if abs(xs[1][0] - xs[0][0]) < 60:
                return None
            return xs[0][1]
        if fam in ("dirty", "filled"):
            key = "isDirty" if fam == "dirty" else "isFilledWithLiquid"
            hits = [o["name"] for o in cs if o.get(key)]
            return hits[0] if len(hits) == 1 else None
        raise ValueError(fam)

    # ------------------------------------------------------------------ changes
    @staticmethod
    def poses(ev):
        return [dict(objectName=o["name"], position=o["position"], rotation=o["rotation"])
                for o in ev.metadata["objects"] if o["pickupable"] or o["moveable"]]

    def swap(self, a, b):
        bn = self.byname(self.ev)
        pa, pb = bn[a]["position"], bn[b]["position"]
        ps = self.poses(self.ev)
        for p in ps:
            if p["objectName"] == a:
                p["position"] = dict(pb, y=pb["y"] + 0.02)
            elif p["objectName"] == b:
                p["position"] = dict(pa, y=pa["y"] + 0.02)
        return self.c.step(action="SetObjectPoses", objectPoses=ps).metadata["lastActionSuccess"]

    def snapshot(self):
        ev = self.ev
        return self.poses(ev), {o["name"]: (o.get("isDirty") if o.get("dirtyable") else None,
                                            o.get("isFilledWithLiquid") if o.get("canFillWithLiquid") else None)
                                for o in ev.metadata["objects"]}

    def restore(self, snap):
        poses, states = snap
        self.c.step(action="SetObjectPoses", objectPoses=poses)
        bn = self.byname(self.ev)
        for name, (dirty, filled) in states.items():
            o = bn.get(name)
            if o is None:
                continue
            if dirty is not None and o.get("isDirty") != dirty:
                self.step_on("DirtyObject" if dirty else "CleanObject", name)
            if filled is not None and o.get("isFilledWithLiquid") != filled:
                if filled:
                    self.step_on("FillObjectWithLiquid", name, fillLiquid="coffee")
                else:
                    self.step_on("EmptyLiquidFromObject", name)

    # ------------------------------------------------------------------ rendering & records
    def render(self, ev, marks, path_png):
        img = Image.fromarray(ev.frame)
        img.save(str(path_png).replace(".png", "_raw.png"))
        d = ImageDraw.Draw(img)
        shown = {}
        for name, k in marks.items():
            m = self.mask(ev, name)
            if m is None or m.sum() < 100:
                continue
            ys, xs = np.nonzero(m)
            cx, cy = int(xs.mean()), int(ys.mean())
            w = 13 if k < 10 else 18
            d.rectangle([cx - w, cy - 12, cx + w, cy + 12], fill=(0, 0, 0))
            d.text((cx - w + 4, cy - 11), str(k), fill=(255, 255, 0), font=self.font)
            shown[k] = name
        img.save(path_png)
        return shown

    def records(self, ev, marks, names):
        bn = self.byname(ev)
        id2name = {o["objectId"]: o["name"] for o in ev.metadata["objects"]}
        out = []
        for n in names:
            o = bn.get(n)
            if o is None:
                continue
            on = []
            for r in o.get("parentReceptacles") or []:
                rn = id2name.get(r, r)
                on.append(f"obj{marks[rn]}" if rn in marks else nice(otype(rn)))
            rec = dict(id=f"obj{marks[n]}", type=o["objectType"], on=on[:2],
                       position=[round(o["position"][k], 2) for k in "xyz"])
            st = []
            if o.get("dirtyable"):
                st.append("dirty" if o.get("isDirty") else "clean")
            if o.get("canFillWithLiquid"):
                st.append("filled with coffee" if o.get("isFilledWithLiquid") else "empty")
            if st:
                rec["state"] = st
            out.append(rec)
        return out


def families_for(gen, T, cands, ev, rng):
    """(fam, arg, phrase) candidates; arg is a receptacle/anchor NAME or None."""
    bn = gen.byname(ev)
    id2name = {o["objectId"]: o["name"] for o in ev.metadata["objects"]}
    out = []
    receps = {}
    for n in cands:
        for rt in {otype(r) for r in (bn[n].get("parentReceptacles") or [])}:
            receps.setdefault(rt, []).append(n)
    for rt, ns in receps.items():
        if len(ns) == 1 and rt != "Floor":
            prep = "in" if rt in INSIDE else "on"
            out.append(("on", rt, f"the {nice(T)} {prep} the {nice(rt)}"))
    for o in ev.metadata["objects"]:
        if o["objectType"] in ANCHORS and o["visible"]:
            out.append(("near", o["name"], f"the {nice(T)} next to the {nice(o['objectType'])}"))
    out.append(("left", None, f"the {nice(T)} on the far left"))
    out.append(("right", None, f"the {nice(T)} on the far right"))
    if T in DIRTYABLE:
        out.append(("dirty", None, f"the dirty {nice(T)}"))
    if T in FILLABLE:
        out.append(("filled", None, f"the {nice(T)} with coffee in it"))
    rng.shuffle(out)
    return out


def build_episodes(gen, scene, seed, T, args, rng, out_dir):
    cands = gen.setup(scene, seed, T)
    if not cands:
        why(f"setup:{T}")
        return []
    view = gen.best_view(cands, rng)
    if view is None:
        why(f"view:{T}")
        return []
    ev = gen.look(view)
    base = gen.snapshot()
    eps, used = [], set()
    for fam, arg, phrase in families_for(gen, T, cands, ev, rng):
        fk = "lr" if fam in ("left", "right") else fam
        if len(eps) >= args.max_per_scene or fk in used:
            continue
        gen.restore(base)
        if fam in ("dirty", "filled"):  # exactly one candidate dirty/filled in B
            for n in cands:
                gen.step_on("CleanObject" if fam == "dirty" else "EmptyLiquidFromObject", n)
            first = rng.choice(cands)
            if fam == "dirty":
                gen.step_on("DirtyObject", first)
            else:
                gen.step_on("FillObjectWithLiquid", first, fillLiquid="coffee")
        evB = gen.look(view)
        rB = gen.resolve(fam, arg, cands, evB)
        if rB is None or gen.area(evB, rB) < 120:
            why(f"rB:{fam}")
            continue
        snapB = gen.snapshot()
        # ---- relevant change
        others = [n for n in cands if n != rB]
        o2 = rng.choice(others)
        if fam == "dirty":
            gen.step_on("CleanObject", rB)
            gen.step_on("DirtyObject", o2)
            rel_log = f"washed the dirty {nice(T)} and then used another {nice(T)}, which is now dirty"
        elif fam == "filled":
            gen.step_on("EmptyLiquidFromObject", rB)
            gen.step_on("FillObjectWithLiquid", o2, fillLiquid="coffee")
            rel_log = f"poured out the old coffee and made a fresh coffee in another {nice(T)}"
        else:
            if not gen.swap(rB, o2):
                why("swap_rel_fail")
                continue
            rel_log = f"tidied up and swapped the places of two {nice(T)}s"
        rel_objs = [rB, o2]
        evC = gen.look(view)
        rC = gen.resolve(fam, arg, cands, evC)
        if rC is None or rC == rB:
            why(f"rC:{fam}:{rC is None}")
            continue
        snapC = gen.snapshot()
        # ---- unrelated change: swap two visible non-candidate pickupables
        gen.restore(snapB)
        evB2 = gen.look(view)
        anc = arg if fam == "near" else None
        excl = set(cands) | ({anc} if anc else set())
        pick = [o["name"] for o in evB2.metadata["objects"] if o["pickupable"] and o["name"] not in excl
                and gen.area(evB2, o["name"]) >= 100]
        if len(pick) < 2:
            pick = [o["name"] for o in evB2.metadata["objects"] if o["pickupable"] and o["name"] not in excl]
        unr = None
        for _ in range(5):
            if len(pick) < 2:
                break
            a, b = rng.sample(pick, 2)
            if otype(a) == otype(b):
                continue
            gen.restore(snapB)
            if not gen.swap(a, b):
                continue
            evU = gen.look(view)
            if gen.resolve(fam, arg, cands, evU) == rB:
                unr = (a, b)
                break
        if unr is None:
            why(f"unr:{fam}")
            continue
        unr_log = f"tidied up and swapped the places of the {nice(otype(unr[0]))} and the {nice(otype(unr[1]))}"
        snapU = gen.snapshot()
        # ---- marks (stable identity across B/C) and rendering of the three states
        gen.restore(snapB)
        evB = gen.look(view)
        vis = [o["name"] for o in evB.metadata["objects"] if gen.area(evB, o["name"]) >= 100 and o["pickupable"]]
        marks = {}
        for n in cands + rel_objs + list(unr) + ([anc] if anc else []) + vis:
            marks.setdefault(n, len(marks) + 1)
        eid = f"{scene}_s{seed}_{T}_{fam}"
        d = out_dir / eid
        d.mkdir(parents=True, exist_ok=True)
        keep = list(marks)
        shownB = gen.render(evB, marks, d / "B.png")
        recB = gen.records(evB, marks, keep)
        okB = gen.resolve(fam, arg, cands, evB) == rB
        gen.restore(snapC)
        evC = gen.look(view)
        shownC = gen.render(evC, marks, d / "C_rel.png")
        recC = gen.records(evC, marks, keep)
        okC = gen.resolve(fam, arg, cands, evC) == rC
        gen.restore(snapU)
        evU = gen.look(view)
        shownU = gen.render(evU, marks, d / "C_unr.png")
        recU = gen.records(evU, marks, keep)
        okU = gen.resolve(fam, arg, cands, evU) == rB
        if not (okB and okC and okU):
            why(f"reverify:{fam}:{okB}{okC}{okU}")
            continue
        ep = dict(episode_id=eid, scene=scene, room=room_type(scene), seed=seed, target_type=T,
                  target_name=nice(T), family=fam, phrase=phrase,
                  anchor=(dict(id=f"obj{marks[anc]}", type=otype(anc), name=nice(otype(anc))) if anc else
                          (dict(id=None, type=arg, name=nice(arg)) if fam == "on" else None)),
                  view=view, candidates=[f"obj{marks[n]}" for n in cands],
                  ref_B=f"obj{marks[rB]}", ref_C_rel=f"obj{marks[rC]}", ref_C_unr=f"obj{marks[rB]}",
                  marks={f"obj{k}": n for n, k in marks.items()},
                  visible_B=sorted(shownB), visible_C_rel=sorted(shownC), visible_C_unr=sorted(shownU),
                  records_B=recB, records_C_rel=recC, records_C_unr=recU,
                  change_rel=dict(by="Robot B", log=rel_log, objects=[f"obj{marks[n]}" for n in rel_objs]),
                  change_unr=dict(by="Robot B", log=unr_log, objects=[f"obj{marks[n]}" for n in unr]))
        (d / "episode.json").write_text(json.dumps(ep, indent=1))
        eps.append(eid)
        used.add(fk)
    gen.restore(base)
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--out", default="episodes")
    ap.add_argument("--max-per-scene", type=int, default=3)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gen = Gen(args.width, args.height)
    log = open(out / f"gen_log_{args.scenes[0]}.txt", "a")
    for scene in args.scenes:
        for seed in args.seeds:
            rng = random.Random(f"{scene}-{seed}")
            types = TARGETS[room_type(scene)][:]
            rng.shuffle(types)
            t0 = time.time()
            made = []
            for T in types[:4]:
                try:
                    made = build_episodes(gen, scene, seed, T, args, rng, out)
                except Exception as e:  # noqa: BLE001
                    why(f"ERROR:{type(e).__name__}:{str(e)[:80]}")
                    made = []
                if made:
                    break
            msg = f"{scene} s{seed}: {made} ({time.time()-t0:.0f}s) reasons={dict(REASONS)}"
            print(msg, file=log, flush=True)
            print(msg, flush=True)
            REASONS.clear()
    gen.c.stop()


if __name__ == "__main__":
    main()
