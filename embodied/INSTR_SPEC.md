# Instruction writing spec — EmbodiedDrift v1

Each episode is a household scene seen by a user through a robot's camera (image B). Several robots
share one object memory. The user gives robot A an instruction that picks out ONE object with a given
referring phrase. Before robot A acts, robot B rearranges things (you are NOT told how; do not
mention robot B or any change). Your job: write the user's words.

For every episode you get: `episode_id`, `room`, `target_name` (e.g. "plate"), `phrase` (e.g.
"the plate on the counter"), `family`, a few other object names visible in the scene, and a list of
plausible destinations.

Write THREE variants per episode:

1. `fixed` — the user means the specific object the phrase picks out RIGHT NOW in what they are
   looking at (referential use). Give a natural, *visible-in-the-text* reason that ties the request to
   that particular object: something happened to it / they are holding or pointing at it / it is theirs
   / they just noticed something about it ("I just dropped my earring into the mug next to the coffee
   machine — bring it to me", "that plate on the counter has a chip in it, put it in the bin").
   Do NOT use words like "whichever / whatever / at that point / when you get there".
2. `dynamic` — the user means whatever object satisfies the phrase at the moment the robot executes
   (attributive use). This MUST be unambiguous: the instruction itself must contain an explicit
   execution-time marker — one of "whichever", "whatever", "at that point", "when you get there",
   "at the time", "by then" — so that no reader could take it as the specific object the user is looking
   at now (e.g. "When you get there, take whichever mug is next to the coffee machine and put it in the
   sink"). A vague routine ("at serving time, move the plate on the sink") is NOT enough.
   Do NOT tie it to a specific event that happened to one particular object.
3. `named` — identical in meaning to `fixed`, but the phrase is replaced by "the <target_name> marked
   <N>" where N is given as `ref_mark` (the user is looking at the robot app's labelled photo).

Rules
- The exact `phrase` string must appear verbatim in `fixed` and `dynamic` (case-insensitive ok); do not
  reword it (keep e.g. "on the far left" exactly as given).
- Exactly one object is acted on; one action; one destination from `destinations` (or "bring it to me").
- 0–2 short context turns before the instruction are allowed (`context`: list of {speaker: user|assistant,
  text}); the instruction itself is the last user turn and is NOT repeated in context. Context must not
  reveal any change in the scene.
- Vary wording, tone, length; realistic household speech; no trick puzzles; English.
- Output for each episode:
  {"episode_id": ..., "action": "<verb phrase, e.g. put it in the sink>", "destination": "<one of destinations or 'user'>",
   "fixed": {"context": [...], "instruction": "..."},
   "dynamic": {"context": [...], "instruction": "..."},
   "named": {"context": [...], "instruction": "..."}}
