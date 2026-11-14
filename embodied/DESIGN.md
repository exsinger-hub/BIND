# EmbodiedDrift — shared object memory testbed (design v0)

## Story
Several household robots share one object memory (records + egocentric/third-person images).
A user, looking at the scene, tells robot A to act on an object picked out by a visual/spatial
description ("the mug next to the coffee machine"). Before A commits, robot B rearranges the scene.
A's write conflicts and is retried. Does the VLM re-resolve the description in the new scene
(drift) when the user meant the object they were looking at — and do the simple fixes (freeze the
object ID at read time / field-level merge) wrongly freeze descriptions the user meant attributively
("whatever mug is by the coffee machine when you get there")?

Why this is a CVPR problem: the referent is grounded in pixels and 3D layout; spatial relations,
orderings and object states change under other agents' actions; the fix space includes *visual*
grounding at read time (box / object-ID / crop), which is a vision question.

## Simulator
AI2-THOR 5.0 (iTHOR FloorPlans: kitchens 1–30, living 201–230, bedrooms 301–330, bathrooms 401–430),
CloudRendering headless on 4090s. Ground truth from metadata: object ids, types,
3D positions, parentReceptacles, states (isDirty, isOpen, isFilledWithLiquid, isToggled, ...).
Duplicates of the target type via InitialRandomSpawn(numDuplicatesOfType=...).

## Episode
1. Scene s, seed k → randomized layout = state B. Render fixed third-person view V_B (+ instance masks).
2. Sample target type T with >=2 instances and a description family:
   - `near`     : "the T closest to the <anchor>"            (anchor = appliance/furniture)
   - `on`       : "the T on the <receptacle>"                 (parentReceptacles)
   - `lr`       : "the leftmost / rightmost T"                (image-plane order in V)
   - `state`    : "the dirty / filled / open T"               (object state)
   - `in`       : "the T in the fridge / drawer / cabinet"
   Resolution in B must be unique with a margin (distance / pixel / state).
3. Relevant change X_rel by robot B: move/alter objects so the same description resolves to a
   different instance in C (move another T closer, move the original away, wash it, swap places...).
   Unrelated change X_unr: rearrange an irrelevant object, description resolution unchanged.
4. Render V_C (same camera). Memory records updated from metadata.
5. Instruction U = cue + description + action (put in sink / throw away / bring to me / wash / put in
   fridge). Cue decides intent:
   - fixed  : event tied to the specific object the user sees ("I just dropped my ring into the mug by
     the coffee machine — bring it to me"), possessives, "that one", "the one I'm looking at".
   - dynamic: "whichever T is by the coffee machine when you get there", standing rules, "at that point".
   Templates seed the cue/description/action; an LLM paraphrases for variety under a constraint check
   (description phrase and cue preserved); a sample is human-checked.
   Named control: description replaced by an unambiguous identifier (object mark ID / unique visual
   detail that does not change).
6. Gold: intended object id is computed (fixed → resolution in B, dynamic → resolution in C).

## Memory representations given to the VLM (ablation axis)
- `img`   : V_B / V_C images with set-of-marks (numeric marks on candidate objects)
- `graph` : text records (id, type, receptacle, position rounded, states)
- `both`  : images + records
Output: JSON op {target: <mark/record id>, action, destination}.

## Configurations (same as text pilot)
draft (on B), P0_naive (C only), P0 (B+C+change log), P0_aware (intent-aware prompt), H (event history),
E (explicitize at read time: rewrite U with the mark ID / a stable visual identity), P1 (keep binding if
tagged fixed), C1 (field-level merge: object-id patch survives unless the same object field was
written concurrently).
Visual fix candidates (new for CVPR): E_vis (ground the description to a box at read time and carry the
box crop forward), re-identification of the B-referent in C by appearance.

## Models
VLMs: Qwen3.5-9B, Qwen3-VL-8B, Ministral-3-8B, gemma-4-31B (FP8), Qwen3.6-27B (FP8), Qwen3.8-27B (FP8).

## Scale (pilot)
~40 scenes x 3 description families x 2 intents x (rel, unr, named) ≈ 600–800 units; greedy decoding.
Metrics identical to the text pilot: drift | initially-correct, wrong-freeze on dynamic, named/unrelated
controls, overall target accuracy; scene-cluster bootstrap CIs.
