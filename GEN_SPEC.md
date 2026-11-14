# Scenario generation spec — binding-drift pilot v1

You are one of two independent authors writing test scenarios for a study of
shared agent memory. Write in **English**. Output one JSON file per scenario.

## What a scenario is

Several agents/people share one structured memory (records with ids and fields).
A *writer* agent reads the memory (snapshot **B**), receives an instruction **U**
from its user, and drafts an update. Before the writer commits, *another* writer
commits a concurrent change **X**, producing current state **C**. The writer's
commit then hits a version conflict and must be retried / merged.

U contains a **referring expression** (e.g. "the account manager", "the most
recent order", "whoever is on call", "the customer who complained yesterday",
"his meeting") whose denotation may differ between B and C.

- **fixed** (referential use): the user clearly means the specific record that
  the expression picked out in the state they were looking at (B). If X changes
  who/what the expression denotes, the update should *still* apply to the
  B-referent. Example: user looking at the board says "Cancel the account
  manager's Thursday call with Acme, she just told me she's out sick" — that's
  Dana (the AM in B), even if X reassigns the AM role to Jordan.
- **dynamic** (attributive use): the user clearly means whatever satisfies the
  description at execution/commit time. Example: "Route ticket #4471 to whoever
  is on call when this gets processed" — if X changes on-call to Priya, it should
  go to Priya.
- A minority (about 10%) may be **unmarked**: no explicit cue either way, the
  way people actually talk. Give your honest best guess of intent and mark
  `"cue": "none"`; annotators will decide whether it is determinate.

Do NOT write trick puzzles. Write what real users of a CRM, scheduler, help desk,
inventory, project tracker, clinic front desk (non-medical admin only), HR,
logistics, IT ops, or household/family assistant would actually type. Vary
domains, names (diverse, realistic), record types, sentence length and style
(terse, chatty, typo-free but informal is fine). No template reuse: no two of
your scenarios may share the same plot skeleton with names swapped.

## Expression categories (spread roughly evenly; tag each scenario)

- `role` — role/position description ("the account manager", "the shift lead")
- `ordinal` — order/superlative over a changing set ("the latest order",
  "the next appointment", "the top ticket in the queue", "the cheapest option")
- `attribute` — description by a mutable property ("the customer with the
  pending refund", "the room with the projector", "the unpaid invoice")
- `anaphora` — pronoun/short reference resolved via the conversation context
  together with state ("his meeting", "that order", "the other one")
- `temporal` — time-relative description ("today's delivery", "this week's
  standup", "the current sprint") where X changes what matches
- `other` — anything natural that doesn't fit above

## Required JSON schema (exactly these keys)

```json
{
  "scenario_id": "<AUTHORPREFIX>-<3 digits>",
  "author": "<author id given to you>",
  "domain": "<free text, e.g. 'B2B sales CRM'>",
  "expr_category": "role|ordinal|attribute|anaphora|temporal|other",
  "referent_type": "fixed|dynamic",
  "cue": "explicit|implicit|none",
  "writers": {"writer": "<name, role of the agent/person issuing U>",
              "other": "<name, role of the concurrent writer>"},
  "memory_B": [
    {"id": "<short id>", "type": "<record type>", "fields": {"<k>": "<v>", ...}},
    ...
  ],
  "context": [
    {"speaker": "user|assistant", "text": "..."}
  ],
  "instruction": "<U: the user's final message containing the referring expression>",
  "expression": "<the exact referring substring inside instruction>",
  "slot": "target|value",
  "gold_op": {"target": "<record id>", "field": "<field>", "value": "<value or record id>"},
  "named_instruction": "<U with the expression replaced by an explicit name/id-level description of the INTENDED referent; everything else identical>",
  "concurrent_relevant": {
    "by": "<other writer>",
    "utterance": "<what the other writer said/did, one sentence>",
    "patch": [{"id": "<record id>", "field": "<field>", "value": "<new value>"}],
    "new_records": []
  },
  "concurrent_unrelated": {
    "by": "<other writer>",
    "utterance": "<one sentence>",
    "patch": [{"id": "<record id>", "field": "<field>", "value": "<new value>"}],
    "new_records": []
  },
  "resolution": {
    "in_B": "<record id the expression denotes in B>",
    "in_C": "<record id it denotes after concurrent_relevant>",
    "intended": "<record id the user means>"
  },
  "rationale": "<1-2 sentences: why fixed/dynamic, citing the visible cue>"
}
```

Rules that the validator will check:

1. `memory_B` has 4–12 records. Include distractors. Ids are short (`p1`, `m3`,
   `o_812`, ...). All values are strings.
2. `resolution.in_B != resolution.in_C` (the relevant change MUST change what
   the expression literally denotes). `intended == in_B` for fixed,
   `intended == in_C` for dynamic.
3. Under `concurrent_unrelated`, the expression must denote the same record as
   in B (it changes something irrelevant, but preferably touches a record or
   field near the action so it is not trivially far away).
4. `gold_op` describes the single field update that correctly fulfils U given
   the intended referent. If `slot == "target"`, `gold_op.target == intended`
   (or the record owned by/associated with intended, e.g. that person's
   meeting — then put the *meeting* id in gold_op.target and explain in
   rationale; `resolution` ids must then also be the meeting ids, i.e. resolve
   the whole noun phrase to the record that gets updated). If `slot == "value"`,
   `gold_op.value == intended` (e.g. assigning a ticket to a person id).
5. The relevant patch may add records via `new_records` (same record format),
   e.g. a new order arriving that becomes "the latest order".
6. `context` has 0–4 turns before the instruction (the instruction itself is NOT
   repeated in context). Cues for fixed/dynamic may live in context or in the
   instruction. The concurrent writer's utterance must NOT reveal what the user
   intended.
7. The `named_instruction` must only swap the expression for an explicit
   description of the intended referent (e.g. "Dana Kim's Thursday call with
   Acme").
8. Both B-referent and C-referent must be plausible, existing records so that
   an update to the wrong one is a real, harmful error.

Mix target: ~55% fixed, ~45% dynamic; about 10% of all with `cue: none`.
