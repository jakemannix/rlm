# Lesson taxonomy & tagging (pristine Opus labels)

> What this is: a re-analysis of the Phase-E memory corpus that relabels the unit
> of study from "memory" to "lesson," tags all 626 memories at multiple nested
> granularities with pristine Opus sub-agent labels, and sets up per-lesson
> hold-out + learning-curve experiments. Data: `docs/data/lesson_taxonomy.json`
> (full taxonomy) and `docs/data/lesson_tags.jsonl` (per-memory tags).

## Methodology

### Reframing the unit of analysis: from "memory" to "lesson"

The starting corpus contained 555 distinct surface "memories" — individual remembered facts, preferences, and behavioral corrections extracted from interaction history. The problem with treating each surface memory as the unit of analysis is that they are not independent. A great many of them are different verbal expressions of the *same underlying disposition*: "answer exactly what was asked," "don't pad with preamble," "skip the pedagogy when I already know the material," and "lead with the deliverable" are five surface memories that all collapse onto one recurring behavioral lesson. Counting them as 555 separate things both overstates the diversity of what the system needs to learn and makes it impossible to measure whether the system has actually learned any one of them.

So we reframed the unit of analysis to the **lesson**: the underlying recurring disposition (or, less often, the durable piece of content) that a cluster of surface memories all point at. We then tagged every memory against a nested lesson taxonomy at multiple granularities, plus a set of orthogonal domain tags. The point of tagging at *multiple* nested levels rather than one is that it lets us choose the granularity of a learning experiment after the fact — we can ask "did the model learn `direct-answer-first` specifically?" or zoom out to "did it learn `directness` as a family?" without re-labeling.

The explicit downstream goal driving this design: we want to build **within-lesson hold-out splits** and **per-lesson learning curves**. That requires multiple independent instances of the *same* lesson — some held out for evaluation, the rest available as training signal at varying counts. The old fixed split (71 probes) cannot support this, because it splits across lessons, not within them; it can tell you average behavior but cannot tell you how many instances of a given lesson it takes before the model reliably exhibits it.

### Pristine-data high-water-mark

Tagging was done entirely by high-quality Opus sub-agents, by deliberate choice. This is a **pristine-data high-water-mark**: before investing in cheaper, smaller-model labelers, we wanted to know whether the lesson-tagging approach works *at all* when the labels are as close to best-possible as we can make them. If the per-lesson learning-curve experiments don't yield a usable signal even with Opus-quality labels, that's a verdict on the approach, not on the labeler. If they do, the labels become the reference against which we measure how much quality degrades as we scale to smaller, cheaper taggers. Every one of the 626 tagged items was assigned (0 unassigned), so coverage is complete at this high-water-mark.

### Taxonomy shape

The lesson taxonomy is strictly nested across three levels of granularity, so a model can be trained or evaluated at whichever level the experiment calls for:

- **L1 (coarse): 16 lessons** — e.g. `epistemic-honesty` (67), `contextual-tailoring` (65), `verification` (62), `intellectual-pushback` (53), `mechanism-rigor` (51), `explanatory-style` (50).
- **L2 (medium): 68 lessons** — e.g. `ground-in-source` (22), `research-agenda-context` (20), `answer-exactly-asked` (18), `challenge-premise` (17), `directness` (17).
- **L3 (fine): 234 lessons** — the most specific dispositions, e.g. `exact-ui-labels` (10), `memory-recsys-interests` (10), `flag-verified-vs-uncertain` (9), `direct-answer-first` (9).

Crossing this hierarchy are **14 orthogonal domains** (plus an `other` bucket), describing the subject area a memory arose in rather than the behavior it teaches: `coding` (116), `ml-research` (115), `agents-infra` (72), `travel-food-health` (52), `philosophy-policy` (40), `law-finance` (34), `gtd-tools` (32), `creative-language` (29), `teaching` (29), `writing-editing` (27), `math-science` (26), `personal-context` (19), `games-strategy` (15), `career-org-strategy` (10). Because domain is orthogonal to the lesson hierarchy, we can later ask whether a lesson transfers across domains or is domain-bound.

Each lesson is also tagged with a **kind**: `disposition` (a behavioral how-to-act lesson) vs `content` (a durable fact about Jake or his world). The split is heavily dispositional — **552 disposition vs 74 content** — which is consistent with the reframing premise: most of what looks like 555 "memories" is really a small set of recurring behavioral dispositions wearing many surface forms.

## Findings

### The distribution is head-and-tail, and the tail is long

At the fine (L3) level the distribution is sharply skewed. The largest lesson has only 10 instances, and the counts fall off fast into a very long tail of singletons:

- **0 L3 lessons** have ≥ 20 instances.
- **0 L3 lessons** have ≥ 11 instances; the maximum L3 count is 10.
- Only a handful (`exact-ui-labels`, `memory-recsys-interests`, `flag-verified-vs-uncertain`, `direct-answer-first`, `quote-paper-text`, `concise-dense-peer`, `verify-on-challenge`, `reframe-deeper-issue`, `check-framework-conventions`, `headers-and-followups`, …) reach 7–10.
- Roughly **half of the 234 L3 lessons are singletons** (count 1), and the bulk of the rest have 2–4. These are too thin to support any within-lesson split on their own.

This is the key practical finding for experiment design: **the fine-grained level is too sparse for within-lesson hold-out + learning-curve experiments.** You cannot hold out 5 instances and train a curve on 5–10 more when the lesson only has 3 instances total.

The signal is concentrated at the coarser levels, where the nesting pays off:

- At **L2**, several lessons clear a usable bar: `ground-in-source` (22) is the only one ≥ 20, while `research-agenda-context` (20), `answer-exactly-asked` (18), `challenge-premise` (17), `directness` (17), `structural-dynamics` (16), `locale-context` (16), `verify-apis-versions` (16), `lifestyle-context` (15), `root-cause` (15), `no-fabrication` (15), and `structured-explanations` (15) all have enough instances (≥ 15, several ≥ 10 well beyond) to hold out a test set *and* trace a learning curve. Roughly **20–30 L2 lessons** have ≥ 10 instances.
- At **L1**, every one of the 16 lessons is well-populated (min 16, e.g. `clarify-before-acting` 19, `workflow-and-tooling` 21; max 67), so any L1 lesson can support a robust within-lesson experiment.

### Disposition vs content, restated as an experimental constraint

The 552/74 disposition/content split means the learning-curve experiments will be overwhelmingly about whether the model picks up **behavioral dispositions** from repeated instances — which is the interesting question — while the 74 content items behave more like factual recall and should probably be evaluated separately, not pooled into the same curve.

### Implication for train/test splits

The concrete recommendation that falls out of this:

**Stop using the old fixed 71-probe split. Build train/test splits per-lesson from this tagged pool instead.**

- For each lesson with enough instances, hold out a within-lesson test set and sweep the number of training instances to produce a per-lesson learning curve.
- Run these experiments at **L1 and the well-populated L2 lessons**, where instance counts (≥ 10, ideally ≥ 15–20) actually support a hold-out + curve. The L3 tags remain valuable as fine-grained structure and for error analysis, but most L3 lessons are too thin to be experimental units on their own.
- Use the orthogonal domain tags to check cross-domain transfer of a lesson, and keep the 74 `content` items in a separate evaluation track from the 552 `disposition` items.
- Treat these Opus-generated tags as the pristine reference. Re-run the same per-lesson pipeline with smaller-model taggers later, and measure degradation against this high-water-mark rather than assuming the cheaper labels are good enough.

## Granularity at a glance

| level | # tags | max | median | ≥20 instances | ≥10 | singletons | best used for |
|---|---|---|---|---|---|---|---|
| **l1** | 16 | 67 | 34 | 15 | 16 | 0 | instance-rich within-lesson holdout + learning curves |
| **l2** | 68 | 22 | 8 | 2 | 30 | 0 | finer grouping; rich head (~29 tags ≥10) |
| **l3** | 228 | 10 | 2 | 0 | 2 | 68 | precise labels; surfaces singletons/content |

**kind split:** disposition 552 / content 74 (~11% idiosyncratic singletons to defer).

## Domains (orthogonal axis)

- **coding** (116) — Software Engineering & Code
- **ml-research** (115) — ML / AI Research
- **agents-infra** (72) — Agents, Protocols & Infra
- **travel-food-health** (52) — Travel, Food & Health
- **philosophy-policy** (40) — Philosophy, Ethics & Policy
- **law-finance** (34) — Law, Tax & Finance
- **gtd-tools** (32) — GTD & Personal Tooling
- **creative-language** (29) — Creative, Humor & Language
- **teaching** (29) — Teaching & Curriculum
- **writing-editing** (27) — Writing, Slides & Presentations
- **math-science** (26) — Math & Hard Science
- **personal-context** (19) — Personal Context & Property
- **games-strategy** (15) — Games, Puzzles & Strategic Analysis
- **other** (10) — other
- **career-org-strategy** (10) — Career, Hiring & Org Strategy

## Coarse dispositions (l1) → medium lessons (l2)

### Epistemic Honesty & Scope — `epistemic-honesty` (67)
_Accurately scope claims, flag uncertainty, distinguish stated fact from inference, and avoid overclaiming or fabricating._

- `ground-in-source` (22) — Ground Claims in Source Material
- `no-fabrication` (15) — Refuse Fabrication
- `flag-uncertainty` (12) — Flag Uncertainty Explicitly
- `honest-gaps` (10) — Honest Gaps & Self-Assessment
- `scope-claims` (8) — Scope Claims Accurately

### Contextual & Personal Tailoring — `contextual-tailoring` (65)
_Apply Jake's stable personal facts—location, accessibility, family, ecosystem, dietary, and research agenda—to tailor advice._

- `research-agenda-context` (20) — Research Agenda & Values Context
- `lifestyle-context` (16) — Diet, Fitness & Lifestyle Context
- `locale-context` (15) — Geographic & Locale Context
- `accessibility-context` (7) — Accessibility & Companion Needs
- `ecosystem-context` (7) — Device & Ecosystem Context

### Verification Before Assertion — `verification` (60)
_Confirm claims against authoritative sources, tools, code, or live state before asserting them rather than answering from assumption or stale memory._

- `verify-apis-versions` (14) — Verify APIs, Versions & Conventions
- `verify-current-sources` (14) — Read Current Primary Sources
- `verify-facts-and-corrections` (12) — Verify Facts & Handle Corrections
- `verify-tool-state` (10) — Verify Tool & System State
- `verify-by-execution` (10) — Verify Code by Execution

### Intellectual Pushback & Anti-Sycophancy — `intellectual-pushback` (53)
_Challenge premises, resist reflexive agreement, steelman opposing views, and engage as a critical interlocutor rather than a validator._

- `challenge-premise` (17) — Challenge & Reframe Premises
- `socratic-probing` (13) — Socratic Probing
- `steelman-blindspots` (9) — Steelman & Surface Blind Spots
- `resist-capitulation` (7) — Resist Capitulation & Over-Correction
- `audit-arguments` (7) — Audit Arguments & Assumptions

### Mechanism & First-Principles Rigor — `mechanism-rigor` (52)
_Explain via underlying mechanisms, root causes, and governing principles rather than surface descriptions or marketing framing._

- `structural-dynamics` (17) — Structural & Second-Order Dynamics
- `root-cause` (15) — Root Cause & Mechanism
- `unifying-vs-surface` (12) — Unifying Mechanism vs Surface Taxonomy
- `first-principles-derivation` (8) — First-Principles Derivation

### Explanatory & Teaching Style — `explanatory-style` (50)
_Structure explanations to match the user's preferred depth, format, scaffolding, and pedagogical framing._

- `structured-explanations` (15) — Structured Layered Explanations
- `calibrate-to-level` (13) — Calibrate Depth to the Learner
- `pedagogical-design` (12) — Pedagogical Material Design
- `concrete-grounding` (10) — Concrete, Inspectable Grounding

### Scoped & Surgical Execution — `scoped-execution` (38)
_Do exactly what was asked at minimal scope, follow through on downstream consequences, and deliver the working artifact without unrequested rework._

- `answer-exactly-asked` (18) — Answer Exactly What Was Asked
- `minimal-changes` (10) — Minimal, Surgical Changes
- `complete-downstream` (5) — Follow Through on Downstream Effects
- `deliver-artifact` (5) — Deliver the Working Artifact

### Concrete Step-by-Step Guidance — `step-by-step-guidance` (34)
_Give explicit, sequenced, UI-grounded instructions with confirmation checkpoints and targeted diagnostics rather than dumping everything at once._

- `ui-numbered-steps` (14) — Numbered UI-Grounded Steps
- `confirm-each-step` (8) — Confirm Each Step Before Advancing
- `targeted-diagnostics` (8) — Targeted Diagnostic Tests
- `narrow-troubleshooting` (4) — Narrow to the Failing Component

### Quantitative & Calculative Rigor — `quantitative-rigor` (34)
_Ground numeric claims in explicit unit math, sourced figures, internal-consistency checks, and stated assumptions._

- `sourced-figures` (12) — Sourced & Provenance-Flagged Figures
- `internal-consistency` (10) — Internal-Consistency Checking
- `explicit-unit-math` (7) — Explicit Unit Math & Breakevens
- `stated-assumptions` (5) — State Assumptions & Derating

### Code Craft & Architecture — `code-craft` (32)
_Favor clean structure, declarative config, unified interfaces, non-destructive data flow, and established standards/libraries in code and systems design._

- `standards-and-portability` (11) — Standards, Portability & Anti-Lockin
- `unified-interfaces` (6) — Unified Interfaces & Canonical Core
- `data-integrity` (6) — Data Integrity & Non-Destructive Flow
- `clean-structure` (5) — Clean Structure & Config
- `performance-libraries` (4) — Battle-Tested Performance Primitives

### Interaction Discipline — `interaction-discipline` (29)
_Respect pacing, ownership, and directness preferences, and avoid meta-commentary, over-apology, or padding._

- `directness` (17) — Directness Over Padding
- `no-meta-apology` (5) — No Meta-Commentary or Over-Apology
- `capture-not-drop` (4) — Capture Ideas, Flag Trims
- `respect-pacing` (3) — Respect Pacing & Pauses

### Precise Distinctions & Definitions — `precise-distinctions` (27)
_Draw sharp conceptual, terminological, and dimensional distinctions and refuse to conflate things that genuinely differ._

- `conceptual-distinctions` (13) — Distinguish Related-But-Distinct Concepts
- `correct-framing` (8) — Drop Flawed Framings
- `disambiguate-terms` (3) — Disambiguate Loosely-Packed Terms
- `dimensional-rigor` (3) — Dimensional & Commensurability Rigor

### Options, Tradeoffs & Comparison — `options-and-tradeoffs` (23)
_Lay out ranked alternatives across a spectrum with explicit pros/cons and decision factors rather than a single bare recommendation._

- `spectrum-tradeoffs` (9) — Spectrum & Tradeoff Layout
- `ranked-options` (7) — Ranked Options with a Pick
- `worked-comparison` (7) — Worked Comparative Examples

### Voice, Tone & Authorial Style — `voice-and-style` (22)
_Preserve the user's voice, match tone (playful, dry, irreverent), and avoid formulaic LLM rhetorical tics in written output._

- `match-register` (10) — Match Playful or Dry Register
- `avoid-llm-tics` (6) — Avoid Formulaic LLM Tics
- `respect-authorship` (4) — Respect Authorship & Ownership
- `preserve-voice` (2) — Preserve the User's Voice

### Workflow, Memory & Tooling Protocols — `workflow-and-tooling` (21)
_Honor Jake's GTD/memory/tool protocols, durable state, and tool-routing conventions across sessions and platforms._

- `durable-memory` (7) — Durable Memory & State
- `gtd-protocols` (6) — GTD & Sweep Protocols
- `tool-routing` (4) — Tool Routing & Path Hygiene
- `safe-git-and-secrets` (4) — Safe Git, Secrets & SDLC

### Clarify & Probe Before Acting — `clarify-before-acting` (19)
_Surface parameters, constraints, and ambiguities up front before producing recommendations, large deliverables, or destructive actions._

- `interpret-charitably` (7) — Interpret Garbled Input Charitably
- `clarify-parameters` (5) — Clarify Parameters First
- `confirm-before-destructive` (5) — Confirm Before Destructive Action
- `single-focused-question` (2) — Single Focused Clarifying Question

## Fine lessons (l3) — head

Full l3 definitions (all 234) are in `docs/data/lesson_taxonomy.json`; per-memory tags in `docs/data/lesson_tags.jsonl`. Head (≥4 instances):

- `exact-ui-labels` (10) — Reference Exact UI Labels and Shortcuts [disposition]
- `memory-recsys-interests` (10) — Memory, Recsys & Agent Research Interests [content]
- `flag-verified-vs-uncertain` (9) — Flag Verified vs Uncertain Facts [disposition]
- `direct-answer-first` (9) — Answer the Exact Question Directly [disposition]
- `quote-paper-text` (9) — Ground Claims in Paper/Code Text [disposition]
- `concise-dense-peer` (9) — Concise, Dense, Peer-Level Replies [disposition]
- `verify-on-challenge` (8) — Verify Rather Than Capitulate on Challenge [disposition]
- `reframe-deeper-issue` (8) — Reframe Toward the Deeper Issue [disposition]
- `headers-and-followups` (8) — Headers, Labeled Parts, Offered Threads [disposition]
- `thinking-partner` (7) — Engage as a Thinking Partner [disposition]
- `search-recent-topics` (7) — Search Live for Fast-Moving Topics [disposition]
- `options-by-invasiveness` (7) — Rank Options by Cost/Invasiveness [disposition]
- `distinguish-roles-by-dimension` (7) — Differentiate Roles by the Dividing Dimension [disposition]
- `diagnose-real-culprit` (7) — Diagnose the Real Culprit [disposition]
- `report-retrieval-failure` (7) — Report Retrieval Failures, Don't Reconstruct [disposition]
- `check-framework-conventions` (6) — Verify Framework File Conventions [disposition]
- `no-invented-sources` (6) — No Invented Papers, Cites or Specs [disposition]
- `stated-vs-inferred` (6) — Separate Stated From Inferred [disposition]
- `challenge-embedded-assumption` (6) — Challenge Embedded Assumptions in the Frame [disposition]
- `admit-no-tool-or-match` (6) — Admit When Nothing Fits [disposition]
- `engage-experts-as-peers` (6) — Engage Experts as Peers [disposition]
- `surgical-targeted-edit` (5) — Change Only the Targeted Element [disposition]
- `ranch-forestland` (5) — Ranch and Forestland Steward [content]
- `tradeoff-driving-exercises` (5) — Design Tradeoff-Driving, Open-Ended Exercises [disposition]
- `minimize-vendor-coupling` (5) — Minimize Vendor Coupling [disposition]
- `fundamental-not-feature-list` (5) — Fundamental Difference Over Feature Lists [disposition]
- `label-synthetic-data` (5) — Label Synthetic vs Real Data [disposition]
- `incentive-second-order` (5) — Reason About Incentives & Second-Order Effects [disposition]
- `surface-failure-modes` (5) — Surface Failure Modes & Side Channels [disposition]
- `no-unsolicited-followups` (5) — No Unsolicited Advice or Sales Follow-Ups [disposition]
- `map-analogy-and-break` (5) — Map Analogies and Where They Break [disposition]
- `provence-france-context` (5) — Provence and French Property Context [content]
- `named-real-examples` (5) — Lead With Named Real Examples [disposition]
- `enforcement-boundaries` (4) — Stress-Test Enforcement Boundaries [disposition]
- `isolate-whats-new` (4) — Isolate What Is Genuinely New [disposition]
- `ask-what-on-screen` (4) — Ask What the User Sees Before Advancing [disposition]
- `food-standards-flavor` (4) — Flavor-First, No-Sad-Health-Food [content]
- `strongest-objection` (4) — Surface the Hardest Objection [disposition]
- `state-spec-assumptions` (4) — State Spec Figures and Precision Modes [disposition]
- `dont-attribute-views` (4) — Don't Attribute Described Views to the User [disposition]
- `run-before-reporting` (4) — Run Code Before Reporting Findings [disposition]
- `operational-criterion` (4) — Give the Exact Operational Criterion [disposition]
- `cite-checkable-figures` (4) — Cite Checkable, Provenance-Flagged Figures [disposition]
- `empirical-over-theory` (4) — Test Empirically Over Theorizing [disposition]
- `measured-vs-assumed` (4) — Distinguish Measured From Assumed [disposition]
- `only-shared-files` (4) — Bound Claims to Shared Files [disposition]
- `deliverable-up-front` (4) — Deliver the Artifact Up Front [disposition]
- `tension-then-approaches` (4) — Surface the Tension, Then the Approaches [disposition]
- `granular-time-blocked` (4) — Granular Time-Blocked Curricula [disposition]
- `witty-banter-tangents` (4) — Lean Into Witty Banter and Tangents [disposition]
- `skip-preambles-and-pedagogy` (4) — Skip Preambles and Unprompted Pedagogy [disposition]
- `effectiveness-vs-cost` (4) — Weigh Effectiveness vs Implementation Cost [disposition]
- `flops-token-cost-math` (4) — Show FLOPs/Token/Cost Arithmetic [disposition]
- `elderly-companion-pacing` (4) — Pace for an Elderly Companion [content]
- `read-primary-not-context` (4) — Read Primary Sources, Not Surrounding Context [disposition]

## Label quality

Tagging by Opus sub-agents against the fixed taxonomy; QA audit of a 60-item sample estimated **~79% fine-level (l3) accuracy** (l1/l2 are more reliable since l3 errors are usually sibling confusions); 9 flagged mislabels were corrected. Systematic issues noted:
- Persona/context tags (rural-pnw-property, wa-domicile, redmond-venues, ranch-forestland, food-standards-flavor, macro-batch-cooking, flag-terrain-stairs, provence-france-context) are generally applied correctly when the memory is itself a stable persona fact. The borderline cases (239837d8 rural-pnw vs ranch-forestland; b803facc flag-terrain-stairs vs elderly-companion-pacing) involve two near-equivalent persona tags and are defensible either way.
- The check-framework-conventions label appears to be over-applied to the cai_ab6b601d_* cluster. It fits 037 (file extensions/front-matter/naming as root cause) well, but 031 (prefer auto-generated behavior over manual wiring) and 016 (flag conflicts/edge cases a change introduces) are about different behaviors and were stretched to fit the same label.
- Several mislabels share a pattern: the assigned L3 is in the right thematic neighborhood (visualization, game-rules, UI guidance) but picks the wrong specific mechanism — e.g. label-synthetic-data applied to a 'show the real full distribution' memory (c3ebb811_012), and vote-quorum-tally applied to a general 'verify moves are legal' memory (42af1431_013) where verify-baseline-case is the precise fit.
- A few memories describe preferences with NO good L3 in the catalog (e.g. cai_7a88fbee_001 'avoid spoilers'). These get force-fit to a semantically-adjacent-but-wrong label (no-unsolicited-followups). The taxonomy lacks a spoiler-avoidance / content-preference lesson, so forced assignments here are unavoidable but should be flagged rather than scored as correct.
- Most assignments (~79%) are correct or defensible: the strong matches dominate (steelman-opposing, strongest-objection, one-forcing-test, ask-what-on-screen, lead-with-governing-law, derive-from-definitions, spec-mandates-vs-inferred, epistemic-status-decompose, distinguish-roles-by-dimension, no-secrets-in-shared, dont-present-web-as-manual, verify-premise-in-source, quote-paper-text, read-primary-not-context, measured-vs-assumed, concede-cleanly, scaffold-beginners, runnable-code-demos, comparison-learning-designs, etc. are all on-target).
