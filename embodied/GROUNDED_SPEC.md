# Memory-grounded intent spec — EmbodiedDrift v2 (plan A)

Setting as in INSTR_SPEC.md: a user asks household robot A to act on one object picked out by `phrase`
(e.g. "the mug on the coffee machine"). Several robots share one memory that also stores NOTES.

Here the user's words are the SAME in two conditions; what the user means is decided by a note that
already sits in the shared memory:

- `utterance`: the user's request. It must contain `phrase` verbatim and point to earlier context in a
  natural way ("like I mentioned", "as we discussed", "the usual", "per the note", "you know the one", ...)
  WITHOUT deciding the reading by itself. Forbidden words in the utterance: whichever, whatever, whoever,
  at that point, when you get there, at the time, by then, any, currently, then, that one, this one, every,
  each, always, rule, routine. One action, one destination from `destinations` (or "user").
- `ref_note`: a note in the shared memory that ties the request to ONE specific object, written as it was
  logged earlier, e.g. "[08:55] Mia: my ring slipped into <OBJ> — please get it to the sink later".
  It MUST contain the literal placeholder <OBJ> exactly once (the program inserts the object id), must make
  the same request/purpose as the utterance, and must not contain `phrase`.
- `att_note`: a standing rule stored in the shared memory that makes the request apply to whatever object
  fits the description, e.g. "House rule (set by Mia): during the evening reset, the mug on the coffee
  machine goes to the sink." It MUST contain `phrase` verbatim (or its relation part, e.g. "on the
  coffee machine" + the object noun), must NOT contain <OBJ> or any object id, and must make the same
  request as the utterance.

Both notes must be things a household would plausibly log. Vary names, times and wording.

Output ONLY a JSON array, one object per input item, in input order:
{"episode_id": "...", "action": "...", "destination": "...", "utterance": "...", "ref_note": "...", "att_note": "..."}
