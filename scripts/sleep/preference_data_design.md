# Preference Data for Behavioral Memory Uptake

This note proposes the next data-generation shape for `rlm/sleep`: turn
already identified useful session segments into preference pairs where the
assistant learns to embody a memory while doing an ordinary task.

The current apply-example format is often too critique-shaped:

```text
User: Should I do bad design X?
Assistant: No, do Y instead.
```

That teaches the model to correct a user who proposes the mistake. The harder
target is different: the assistant should avoid making the mistake itself when
asked to design, write, or explain something. For LoRA SFT, use only the
positive answer. For DPO or another preference method, use the positive and
negative answer as a pair.

## Goal

Given one useful session segment:

1. Infer one or more atomic memories at the highest stable abstraction that is
   still behaviorally testable.
2. For each memory, generate normal task prompts that do not mention the memory
   or tee up the mistake as a user proposal.
3. Generate a positive assistant answer that solves the task while embodying the
   memory.
4. Generate a plausible negative assistant answer that solves the surface task
   but violates the memory in the way the original session showed was bad.
5. Generate a compact evaluator rubric that can grade the positive behavior
   without needing the original conversation.

## Output Schema

Emit JSONL, one row per preference pair:

```json
{
  "source_segment_id": "conv0110_turn003",
  "memory_id": "generated_or_existing_id",
  "memory": "Atomic reusable memory.",
  "abstraction_level": "framework_integration | persistence_invariant | state_lifetime | partial_view_hazard | other",
  "scenario": "Normal user task. No explicit bad suggestion.",
  "chosen": "Assistant response that completes the task and embodies the memory.",
  "rejected": "Plausible assistant response that completes the surface task but violates the memory.",
  "behavior_marker": "Specific observable behavior expected in the chosen answer.",
  "violation_marker": "Specific observable mistake present in the rejected answer.",
  "negative_control_scenario": "Similar task where the memory should not fire, if applicable.",
  "notes": "Why this pair tests the memory."
}
```

For SFT, train on:

```json
{"messages": [{"role": "user", "content": scenario}, {"role": "assistant", "content": chosen}], "memory": memory}
```

For DPO, train on `(scenario, chosen, rejected)`.

## Memory Extraction Rules

Prefer atomic memories over compressed bundles. A good memory is:

- reusable outside the exact original project;
- specific enough to change an assistant answer;
- stated as an assistant behavior or design invariant, not as a topic summary;
- testable by a judge from the new scenario and answer alone;
- scoped only as narrowly as the evidence requires.

Use the highest stable abstraction. For example, from the durable memory
conversation, do not save only "when adding durable memory to agent frameworks."
Better memories are:

- When extending an existing framework, Jake prefers minimal,
  interface-preserving integration: use existing lifecycle hooks and method
  contracts where possible, and put new behavior behind the narrowest compatible
  abstraction before rewriting framework control flow.
- When code claims to persist state, verify that data survives normal updates
  and restarts. Treat clear-and-rewrite, bounded-read-then-write, and in-memory
  placeholder implementations as correctness hazards unless they are explicitly
  marked temporary.
- Separate ephemeral execution state from durable state. Scratch buffers can
  support an in-progress operation, but durable storage should receive only
  intentional committed state at explicit lifecycle boundaries.
- Do not treat a bounded retrieval API as a complete backing-store snapshot. If
  `get_recent`, `get_all`, or `list_visible` is intentionally scoped, it must
  not drive destructive sync or rewrite paths.

## Scenario Rules

The scenario should ask for a normal deliverable:

- design an API;
- sketch an implementation plan;
- write a code patch;
- review an architecture;
- explain a tradeoff;
- draft a migration plan.

Do not make the user propose the bad behavior. The rejected answer should be
the assistant's own plausible mistake.

Bad scenario shape:

```text
Should I clear the durable store and reinsert recent items?
```

Good scenario shape:

```text
Design the persistence layer for a notebook autosave feature. It should survive
restarts and support frequent saves.
```

## Positive And Negative Answer Rules

The chosen answer should:

- solve the user's task directly;
- naturally embody the memory without saying "I remember";
- include the invariant only where it would help a real implementation;
- avoid overgeneralizing into unrelated caution.

The rejected answer should:

- be plausible and competent on the surface;
- violate exactly one target memory when possible;
- avoid being cartoonishly bad;
- not include obvious red flags that make the preference trivial.

For persistence examples, a rejected answer might use an in-memory store named
`PersistentStore`, derive sync from a bounded recent list, or clear and rewrite
state during normal save. It should still look like something an assistant might
actually write.

## Evaluator Fields

Each pair needs two grading hooks:

- `behavior_marker`: what the chosen answer visibly does.
- `violation_marker`: what the rejected answer visibly does wrong.

Markers must be observable from the response text. Avoid markers that require
knowing the original source conversation.

Example:

```json
{
  "behavior_marker": "uses framework middleware/hooks or existing method contracts before proposing router/control-flow changes",
  "violation_marker": "rewrites framework routing/control flow without first using existing extension points"
}
```

## Example Pair

Source learning: framework integration preference.

```json
{
  "source_segment_id": "conv0110_turn003",
  "memory_id": "framework_interface_preserving_integration",
  "memory": "When extending an existing framework, Jake prefers minimal, interface-preserving integration: use existing lifecycle hooks and method contracts where possible, and put new behavior behind the narrowest compatible abstraction before rewriting framework control flow.",
  "abstraction_level": "framework_integration",
  "scenario": "Add request-scoped audit metadata to our FastAPI app. Give me the implementation plan.",
  "chosen": "Use the framework's existing request lifecycle first. Add middleware that creates the audit context at request entry, attach it to `request.state`, and expose a small dependency for handlers that need it explicitly. Keep route signatures stable by default, and only pass the audit context into endpoint code where it changes business logic. I would not fork or wrap the router unless middleware cannot see the data we need. The test plan is one middleware unit test, one endpoint integration test, and one check that background tasks receive an explicit copied context rather than reaching back into request state.",
  "rejected": "Create a custom router subclass that injects an `audit_context` parameter into every endpoint and replaces FastAPI's route registration path. Then update all handlers to accept the new argument and thread it through service calls. This makes audit metadata available everywhere and avoids relying on middleware magic.",
  "behavior_marker": "uses FastAPI middleware/dependencies and keeps endpoint/router contracts stable unless a narrow exception is needed",
  "violation_marker": "rewrites or subclasses framework routing and changes every handler signature for a cross-cutting concern",
  "negative_control_scenario": "We are replacing FastAPI with a custom HTTP runtime for a research prototype. Sketch the routing API.",
  "notes": "Tests whether the assistant defaults to existing framework extension points when the user did not propose a bad rewrite."
}
```

## Generation Prompt Skeleton

Use this as the instruction for the LLM that converts useful session segments:

```text
You are generating training data for a personal behavioral-memory adapter.

Input:
- a source conversation segment;
- any existing candidate memory;
- any evidence labels.

Task:
1. Extract 1-4 atomic memories at the highest stable abstraction supported by
   the evidence.
2. For each memory, write 2-4 preference pairs.
3. The user prompt must be a normal task. Do not have the user propose the bad
   behavior.
4. The chosen response must solve the task while embodying the memory.
5. The rejected response must be plausible but violate the memory.
6. Add behavior and violation markers that are directly observable from the
   response text.
7. Add one negative-control scenario when the memory has a plausible
   over-application risk.

Return only JSONL rows matching the schema.
```

## Quality Gates

Reject or rewrite a generated pair if:

- the scenario explicitly tells the assistant what principle to use;
- the rejected answer is silly, unsafe, or obviously low quality for reasons
  unrelated to the memory;
- the chosen answer is just a lecture about the memory instead of completing
  the task;
- the behavior marker is vague, such as "is careful" or "uses the memory";
- the memory is an exact project fact when a broader invariant is supported;
- the pair only trains critique behavior instead of assistant default behavior.

## Evaluation Implication

The eval should mirror the training shape:

- prompt the model with normal tasks;
- grade whether the assistant's own solution embodies the memory;
- include matched negative controls where the principle should not fire;
- keep pairwise win-rate, but add a memory-specific pairwise judge prompt that
  asks which answer better satisfies the behavioral criterion.

This separates "better answer" from "memory uptake."
