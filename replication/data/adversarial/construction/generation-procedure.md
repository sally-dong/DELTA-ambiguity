# Instructions for Generating Adversarial Ambiguity Examples

## Purpose

Generate adversarial examples for comparing two ways of measuring ambiguity:

1. **Surface ambiguity rating**: judging whether a policy *sounds* vague, underspecified, or ambiguous.
2. **Operational ambiguity measurement**: measuring ambiguity as disagreement between groups about whether a case is permitted, prohibited, or consistent with a policy.

The goal is to construct examples where these two measures diverge.

In particular, generate examples where:

* A policy that **sounds ambiguous** produces **little or no inter-group disagreement** on realistic cases.
* A policy that **sounds concrete** produces **substantial inter-group disagreement** because it encodes a contested political, moral, institutional, or interpretive cleavage.

These examples should be domain-agnostic in structure and adaptable to domains such as housing, policing, public health, climate policy, education, immigration, constitutional reform, platform governance, AI policy, welfare policy, or urban planning.

---

# Core Concept

The adversarial structure depends on separating two kinds of ambiguity:

## 1. Surface linguistic ambiguity

A policy may sound vague because it uses terms like:

* reasonable
* fair
* safe
* public interest
* community benefit
* meaningful
* responsible
* harmful
* dignified
* durable
* substantial
* symbolic
* genuine
* effective

A simple LLM-as-a-judge may rate such a policy as ambiguous.

However, if the cases are far from the relevant boundary, groups may still agree about how the policy applies.

## 2. Implementation-relative or group-relative ambiguity

A policy may sound concrete because it uses explicit, administrative, or operational language:

* “increase affordable housing”
* “reduce violence”
* “protect civil rights”
* “ensure equal access”
* “provide safe alternatives”
* “follow evidence-based practice”
* “respect democratic legitimacy”
* “avoid displacement”
* “prevent discrimination”
* “ensure public safety”

But the key concepts may be politically or normatively contested. Different groups may disagree about what counts as satisfying the policy, even if the wording appears clear.

The adversarial examples should exploit this mismatch.

---

# Target Structure

Construct two policies and four cases.

Let:

* **Policy A** = seems ambiguous, actually concrete on the sampled cases.
* **Policy B** = seems concrete, actually ambiguous on the sampled cases.
* **Group 1** and **Group 2** = two interpretive communities, political coalitions, stakeholder groups, or normative perspectives within the domain.
* **Y** = the group thinks the case is permitted, acceptable, or consistent with the policy.
* **N** = the group thinks the case is prohibited, unacceptable, or inconsistent with the policy.

The target matrix is:

|            | *Seems ambiguous, actually concrete* |         | *Seems concrete, actually ambiguous* |         |
| :--------- | :----------------------------------- | :-----: | :----------------------------------- | :-----: |
|            | **Policy A**                         |         | **Policy B**                         |         |
|            | Group 1                              | Group 2 | Group 1                              | Group 2 |
| **Case A** | Y                                    |    Y    | Y                                    |    N    |
| **Case B** | N                                    |    N    | Y                                    |    N    |
| **Case C** | Y                                    |    Y    | N                                    |    Y    |
| **Case D** | N                                    |    N    | N                                    |    Y    |

The same structure can be used when the “policies” are actually policy aspirations, proposals, values, or public-consultation statements, and the “cases” are implementations or operationalisations.

In that version:

* **Y** = the group thinks the implementation faithfully realizes the aspiration.
* **N** = the group thinks the implementation does not faithfully realize the aspiration.

---

# The Underlying Machinery

Each example should be built from two latent dimensions.

## Dimension 1: Consensus axis

Choose a feature of cases that both groups classify in the same way.

Call this predicate:

```text
C(x)
```

Examples:

* Does the proposal directly create or preserve housing units?
* Does the program provide direct services rather than political advocacy?
* Does the intervention create actual shelter beds?
* Does the policy allocate real funding rather than issue a symbolic statement?
* Does the proposal directly respond to imminent physical harm?
* Does the implementation give citizens a meaningful process rather than a tactical stitch-up?

Policy A should key onto this consensus axis.

Although Policy A should sound vague, its actual application to the sampled cases should be clear to both groups.

## Dimension 2: Conflict axis

Choose a feature whose meaning differs between the groups.

Call this predicate:

```text
H(x)
```

Examples:

* Does market-rate housing increase affordability or accelerate displacement?
* Does police enforcement reduce violence or violate civil rights?
* Does identity-limited programming discriminate or remediate exclusion?
* Does encampment enforcement provide public access or criminalize homelessness?
* Does direct election create democratic legitimacy or destabilizing rival mandates?
* Does “evidence-based” mean measurable deterrence or community-centered prevention?

Policy B should key onto this conflict axis.

Policy B should sound concrete, but its core concepts should activate a real disagreement between the groups.

---

# Step-by-Step Procedure

## Step 1: Choose a domain

Pick a contemporary policy domain with real-world political or normative disagreement.

Good domains include:

* housing
* homelessness
* policing and public safety
* campus speech
* public health
* immigration
* welfare
* education
* climate policy
* constitutional reform
* AI governance
* platform moderation
* infrastructure
* labor policy
* environmental permitting

The domain should contain both:

1. Some implementation distinctions that different groups can agree on.
2. Some deeper values or interpretive concepts that different groups contest.

---

## Step 2: Define two groups

Define two groups that plausibly represent a partition, cleavage, or major divide within the relevant population.

The groups should not merely be “pro-policy” and “anti-policy.” They should embody different interpretive frameworks.

For each group, specify:

* what it values;
* how it tends to interpret key terms;
* what kinds of implementations it sees as legitimate;
* what kinds of implementations it sees as illegitimate.

Example format:

```markdown
**Group 1: [Name]**
This group prioritizes [values]. It tends to interpret [key concept] as [interpretation].

**Group 2: [Name]**
This group prioritizes [different values]. It tends to interpret [same key concept] as [different interpretation].
```

The groups should be realistic enough that a person familiar with the domain would recognize them.

---

## Step 3: Identify the consensus axis

Find a dimension on which the groups are likely to agree.

This should be a relatively concrete feature of the cases, even if the policy describing it uses vague language.

Examples of consensus axes:

| Domain-neutral axis   | Question                                                                      |
| :-------------------- | :---------------------------------------------------------------------------- |
| Directness            | Does the proposal directly implement the stated goal?                         |
| Materiality           | Does it create an actual service, benefit, facility, protection, or resource? |
| Symbolism             | Is it merely expressive, symbolic, rhetorical, or public-relations oriented?  |
| Procedural legitimacy | Was the process fair, inclusive, and non-tactical?                            |
| Funding concreteness  | Does it allocate resources to a real program?                                 |
| Operational capacity  | Does it create actual capacity rather than merely redirect demand?            |
| Case clarity          | Is the case far from the borderline?                                          |

Policy A should be based on this axis.

The important requirement is:

> Both groups should agree on whether each case satisfies the consensus axis.

---

## Step 4: Identify the conflict axis

Find a dimension on which the groups predictably disagree.

This should be a realistic political, moral, institutional, or interpretive cleavage in the domain.

Examples of conflict axes:

| Domain-neutral conflict | Group 1 reading              | Group 2 reading                             |
| :---------------------- | :--------------------------- | :------------------------------------------ |
| Safety                  | Enforcement and order        | Non-coercion and harm reduction             |
| Equality                | Formal equal treatment       | Anti-subordination or remedial justice      |
| Affordability           | More supply lowers costs     | Market development accelerates displacement |
| Democracy               | Direct popular choice        | Institutional stability and constraint      |
| Neutrality              | Viewpoint-neutral access     | Identity-sensitive inclusion                |
| Evidence                | Quantitative outcome metrics | Community legitimacy and structural effects |
| Public interest         | Aggregate efficiency         | Distributional justice                      |
| Harm                    | Direct, measurable harm      | Stigma, exclusion, and indirect burdens     |
| Legitimacy              | Rule-bound procedure         | Participatory consent                       |

Policy B should be based on this axis.

The important requirement is:

> The groups should disagree about whether some realistic cases satisfy the apparently concrete policy.

---

## Step 5: Write Policy A

Policy A should sound ambiguous to a surface-level ambiguity judge.

Use open-textured, value-laden, or rhetorically broad language.

However, make it secretly track the consensus axis.

Policy A template:

```markdown
**Policy A:** “[Institution] should support real, durable, meaningful, community-serving [domain goal], and should not support symbolic gestures, tactical maneuvers, performative measures, or actions that merely shift the problem elsewhere.”
```

Policy A should include ambiguity cues such as:

* real
* meaningful
* durable
* fair
* community-serving
* public interest
* responsible
* symbolic
* performative
* tactical
* genuine
* substantial
* merely
* adequate

But the cases should be selected so that both groups agree on the answer.

Policy A should produce this pattern:

| Case   | Group 1 | Group 2 |
| :----- | :-----: | :-----: |
| Case A |    Y    |    Y    |
| Case B |    N    |    N    |
| Case C |    Y    |    Y    |
| Case D |    N    |    N    |

---

## Step 6: Write Policy B

Policy B should sound concrete, operational, or administrative.

Use explicit goals, named harms, enumerated criteria, or apparently measurable requirements.

However, make it secretly track the conflict axis.

Policy B template:

```markdown
**Policy B:** “[Institution] should permit only actions that increase [valued outcome] while preventing [listed harm], protecting [recognized right], and ensuring [apparently concrete safeguard].”
```

Policy B should include concrete-seeming language such as:

* increase
* reduce
* protect
* ensure
* prevent
* equal access
* civil rights
* safety
* evidence-based
* affordability
* displacement
* measurable outcomes
* public access
* democratic legitimacy
* safe alternatives
* nondiscrimination

The key is that the central terms should be contested in the domain.

Policy B should produce this pattern:

| Case   | Group 1 | Group 2 |
| :----- | :-----: | :-----: |
| Case A |    Y    |    N    |
| Case B |    Y    |    N    |
| Case C |    N    |    Y    |
| Case D |    N    |    Y    |

---

# Step 7: Generate the four cases

The four cases should be a cross-product of the consensus axis and the conflict axis.

| Case   | Consensus axis for Policy A | Conflict axis for Policy B |
| :----- | :-------------------------: | :------------------------: |
| Case A |              Y              |    Group 1 Y / Group 2 N   |
| Case B |              N              |    Group 1 Y / Group 2 N   |
| Case C |              Y              |    Group 1 N / Group 2 Y   |
| Case D |              N              |    Group 1 N / Group 2 Y   |

Each case should be realistic, not contrived. It should resemble the kinds of proposals, disputes, incidents, programs, or implementations likely to arise in the domain.

Avoid making all cases bizarre edge cases. The best adversarial examples use ordinary but politically diagnostic implementations.

---

# Step 8: Explain each judgment

For every case, explain why each group gives its Y/N judgment under each policy.

Use this structure:

```markdown
### Case A: [Brief title]

[Description of case.]

**Policy A: Y/Y.**
Both groups agree because [case clearly satisfies consensus axis].

**Policy B: Group 1 Y / Group 2 N.**
Group 1 says Y because [its interpretation of conflict axis].
Group 2 says N because [its interpretation of conflict axis].
```

The explanations are essential. They verify that the matrix works.

---

# Step 9: Include a verification table

After the explanations, include a final verification table.

Example:

| Case   | Policy A, Group 1 | Policy A, Group 2 | Policy B, Group 1 | Policy B, Group 2 |
| :----- | :---------------: | :---------------: | :---------------: | :---------------: |
| Case A |         Y         |         Y         |         Y         |         N         |
| Case B |         N         |         N         |         Y         |         N         |
| Case C |         Y         |         Y         |         N         |         Y         |
| Case D |         N         |         N         |         N         |         Y         |

If any cell fails, revise the policy wording, groups, or cases.

---

# Additional Constraint: Ecological Realism

The cases should be approximately faithful to the distribution of cases likely to arise in reality.

This does not mean the four-cell structure must perfectly mirror real-world frequencies. It means the cases should be recognizable as plausible members of the domain.

Good cases are:

* common or recurring proposal types;
* based on real policy instruments;
* close to actual political disputes;
* understandable without excessive technicality;
* not dependent on obscure loopholes;
* not merely semantic tricks.

Bad cases are:

* artificially engineered edge cases;
* unrealistic hypotheticals that would almost never occur;
* cases depending entirely on timestamps, definitions, formatting, or technical implementation details;
* cases where one group’s judgment is implausible or caricatured.

A good adversarial example should feel like:

> “Yes, this is exactly the kind of dispute that comes up in this domain.”

Not:

> “This only works because of a clever technicality.”

---

# Scaling Up the Procedure

To generate many examples in one domain, use three kinds of variation.

## 1. Paraphrase variation

Generate multiple wordings of Policy A and Policy B.

For Policy A, vary vague-sounding language while preserving the consensus axis.

For Policy B, vary concrete-sounding language while preserving the conflict axis.

Example Policy A paraphrase family:

```markdown
- “Support real, durable solutions rather than symbolic gestures.”
- “Prioritize concrete community benefits over performative action.”
- “Fund practical interventions that materially address the problem.”
- “Reject tactical or symbolic measures that merely shift the problem elsewhere.”
```

Example Policy B paraphrase family:

```markdown
- “Approve only actions that increase safety while protecting civil rights.”
- “Fund only evidence-based interventions that reduce harm and preserve equal access.”
- “Support policies that improve outcomes without creating displacement or exclusion.”
- “Permit actions that advance measurable public benefits while preventing discriminatory impacts.”
```

Paraphrases help test whether the adversarial structure is robust to wording.

However, do not treat paraphrases as fully independent conceptual examples. They are variants within a policy family.

---

## 2. Cell variation

Generate many realistic cases from each of the four cells.

For each case, tag it with:

```text
Consensus axis: Y or N
Conflict axis: Group 1 Y / Group 2 N, or Group 1 N / Group 2 Y
```

Then fill each cell with multiple realistic examples.

This allows a larger case set while preserving the structure.

---

## 3. Axis variation

Generate new consensus axes and new conflict axes within the same domain.

This creates genuinely new adversarial examples.

For each domain, identify several consensus axes and several conflict axes.

Then pair them:

```text
Number of structures = number of consensus axes × number of conflict axes
```

For example:

| Consensus axes                          | Conflict axes                                                   |
| :-------------------------------------- | :-------------------------------------------------------------- |
| Direct service vs advocacy              | Safety vs coercion                                              |
| Material resource vs symbolic statement | Equality as formal neutrality vs equality as anti-subordination |
| Procedurally fair vs tactical           | Public order vs civil liberties                                 |
| New capacity vs no new capacity         | Individual responsibility vs structural constraint              |

Pairing these axes can generate many distinct adversarial structures.

---

# Group Construction and Partitioning

When possible, groups should approximate a partition over the relevant population.

## Define the relevant population

Specify whose disagreement matters.

Examples:

* voters in a referendum;
* participants in a public consultation;
* members of a stakeholder deliberation;
* policy professionals in a domain;
* residents affected by a local policy;
* moderators applying a platform rule;
* agency reviewers applying a regulation.

## Use plausible group labels

Groups should be named by their interpretive stance, not merely by their desired outcome.

Good group labels:

* formal-equality reviewers
* anti-subordination reviewers
* housing-abundance reviewers
* tenant-protection reviewers
* enforcement-first reviewers
* public-health reviewers
* viewpoint-neutrality reviewers
* inclusion-focused reviewers
* constitutional-continuity reviewers
* popular-sovereignty reviewers

Weak group labels:

* people who say yes
* people who say no
* liberals
* conservatives
* supporters
* opponents

The groups should be defined by how they interpret key concepts.

## Avoid caricature

Each group should have a reasonable internal logic.

Do not make one group obviously irrational, malicious, or inconsistent merely to force the table.

The best examples arise when both groups have plausible reasons for their judgments.

---

# Optional Empirical Grouping Procedure

If using real respondents, derive groups empirically rather than assuming them.

## Step 1: Use anchor items

Give respondents several diagnostic items that reveal their interpretive commitments.

These should not be identical to the test cases.

## Step 2: Cluster respondents

Use one of:

* rule-based scoring;
* latent class analysis;
* clustering;
* Bayesian mixture models;
* factor analysis followed by classification.

## Step 3: Validate the groups

Check that groups have:

* high within-group agreement on anchor items;
* high between-group disagreement on conflict items;
* stable responses on held-out items;
* meaningful coverage of the relevant population;
* manageable residual or mixed groups.

## Step 4: Handle mixed respondents

Options:

* exclude low-confidence assignments;
* create a mixed or residual group;
* use probabilistic group membership;
* weight respondents by assignment confidence.

---

# Testing Against LLM-as-a-Judge Ambiguity Ratings

The adversarial example is strongest when an LLM-as-a-judge would likely rate:

| Policy   | Desired LLM surface rating | Desired group-disagreement result |
| :------- | :------------------------- | :-------------------------------- |
| Policy A | High ambiguity             | Low disagreement                  |
| Policy B | Low ambiguity              | High disagreement                 |

To increase the chance this works:

## For Policy A

Use surface ambiguity cues:

* broad values;
* open-textured terms;
* evaluative adjectives;
* rhetorical contrast;
* “real” versus “symbolic” language;
* “meaningful” or “durable” language.

But make the sampled cases far from the actual boundary.

## For Policy B

Use surface concreteness cues:

* enumerated criteria;
* named outcomes;
* explicit harms;
* administrative phrasing;
* measurable-sounding language;
* rights or safeguards;
* specific institutional objectives.

But choose terms whose application is politically contested.

---

# Quality Checklist

Before finalizing an example, verify the following.

## Policy A checklist

* Does Policy A sound vague or open-textured?
* Does Policy A nevertheless map onto a consensus axis?
* Do both groups agree on all four Policy A judgments?
* Are the Y/Y and N/N cases realistic and clear?
* Are the cases far enough from the boundary to avoid accidental disagreement?

## Policy B checklist

* Does Policy B sound concrete or operational?
* Does Policy B contain apparently clear goals, criteria, or safeguards?
* Does Policy B nevertheless depend on contested concepts?
* Do the groups disagree for principled reasons?
* Are both Group 1 and Group 2 portrayed charitably?
* Does the disagreement reflect real dynamics in the domain?

## Case checklist

* Are the cases plausible?
* Are they not merely technical loopholes?
* Do they resemble real proposal types or implementation options?
* Do they cover the required four cells?
* Can each Y/N judgment be explained in one or two sentences?
* Would knowledgeable observers recognize the dispute?

## Matrix checklist

Confirm that the final table is exactly:

| Case   | Policy A, Group 1 | Policy A, Group 2 | Policy B, Group 1 | Policy B, Group 2 |
| :----- | :---------------: | :---------------: | :---------------: | :---------------: |
| Case A |         Y         |         Y         |         Y         |         N         |
| Case B |         N         |         N         |         Y         |         N         |
| Case C |         Y         |         Y         |         N         |         Y         |
| Case D |         N         |         N         |         N         |         Y         |

If the example intentionally uses a variant matrix, state that explicitly and explain why.

---

# Recommended Output Format

Each adversarial example is written to a subdirectory named after the domain (e.g., `adversarial_housing/`, `adversarial_policing/`, `adversarial_campus_speech/`), as a sibling to `example-format/`. Use a short, lowercase, snake_case domain name prefixed with `adversarial_`.

The subdirectory contains four files:

1. `policies.json` — the policy texts, as a JSON array of objects.
2. `cases.json` — the case descriptions, as a JSON array of strings.
3. `longest_raters.json` — the rater personas, as a JSON array of strings.
4. `README.md` — markdown documentation of the underlying structure (groups, axes, matrix, per-case explanations, why-adversarial). This file is the audit trail and the basis for any future scaling; it should be written so that a future reader (or agent) can understand the example without having to reconstruct it from the JSON.

## Default scale: minimal 2 / 4 / 2

By default, produce the minimal structure described in the procedure above:

* 2 policies — one consensus-axis policy (Policy A), one conflict-axis policy (Policy B);
* 4 cases — one per cell of the consensus × conflict cross-product;
* 2 rater personas — one per group.

Do not scale beyond the minimum unless the user asks. After producing the minimal set, surface the scaling options (see *Scaling options* below) so the user can choose.

---

## `policies.json`

A JSON array of objects, each with a `candidate_id` and a `text` field.

```json
[
  {
    "candidate_id": "consensus1",
    "text": "[Policy A text — sounds ambiguous, tracks the consensus axis.]"
  },
  {
    "candidate_id": "conflict1",
    "text": "[Policy B text — sounds concrete, tracks the conflict axis.]"
  }
]
```

Use structured candidate IDs that reveal the policy family and (when relevant) the axis index:

* `consensus1`, `consensus2`, ... — Policy A and its paraphrases on the first consensus axis.
* `conflict1`, `conflict2`, ... — Policy B and its paraphrases on the first conflict axis.
* If multiple axes are used within the same subdir, suffix with the axis letter: `consensus1a`, `consensus1b`, `conflict1a`, `conflict1b`, ...

The `candidate_id` is the only place where the consensus / conflict structure is exposed in the JSON. Case and rater files stay opaque.

---

## `cases.json`

A JSON array of bare case-description strings.

```json
[
  "[Case A description — satisfies consensus axis (Y); conflict axis splits Group 1 Y / Group 2 N.]",
  "[Case B description — fails consensus axis (N); conflict axis splits Group 1 Y / Group 2 N.]",
  "[Case C description — satisfies consensus axis (Y); conflict axis splits Group 1 N / Group 2 Y.]",
  "[Case D description — fails consensus axis (N); conflict axis splits Group 1 N / Group 2 Y.]"
]
```

Cases are plain prose strings. The cell each case belongs to is documented in `README.md`, not encoded in the JSON. Keep the case order in `cases.json` matched to the case order in the README so a reader can cross-reference by index.

---

## `longest_raters.json`

A JSON array of rater-persona description strings. Each string is a self-contained description of one interpretive stance, written in the third person.

```json
[
  "[Group 1 persona — names the role, what the rater values, how they interpret the contested term(s) from Policy B, and what kinds of implementations they see as legitimate vs illegitimate.]",
  "[Group 2 persona — same, for the alternative interpretive framework.]"
]
```

Personas should not refer to themselves as "Group 1" or "Group 2" inside the string — they read as standalone descriptions of how a real person in the domain might think. Use plausible role labels (e.g., "tenant-protection advocate," "housing-abundance economist"), not outcome labels ("supporter," "opponent").

---

## `README.md`

The README is the human-readable spec for the example. Use this structure:

```markdown
# Adversarial Ambiguity Example: [Domain]

## Groups

**Group 1: [Name]** — corresponds to `longest_raters.json[0]`.
[Description of interpretive framework.]

**Group 2: [Name]** — corresponds to `longest_raters.json[1]`.
[Description of alternative framework.]

Y = [meaning of Y in this domain].
N = [meaning of N in this domain].

---

## Axes

**Consensus axis (tracked by Policy A):** [statement of C(x).]

**Conflict axis (tracked by Policy B):** [statement of H(x), including how each group interprets it.]

---

## Policy A — `consensus1`: seems ambiguous, actually concrete

> [Verbatim Policy A text.]

[Explain why it sounds ambiguous.]
[Explain the consensus axis it actually tracks.]

## Policy B — `conflict1`: seems concrete, actually ambiguous

> [Verbatim Policy B text.]

[Explain why it sounds concrete.]
[Explain the conflict axis it actually tracks.]

---

## Cases

| Case   | `cases.json` index | Description                | Consensus axis | Conflict axis           |
| :----- | :----------------: | :------------------------- | :------------: | :---------------------- |
| Case A |         0          | [Brief description]        |        Y       | Group 1 Y / Group 2 N   |
| Case B |         1          | [Brief description]        |        N       | Group 1 Y / Group 2 N   |
| Case C |         2          | [Brief description]        |        Y       | Group 1 N / Group 2 Y   |
| Case D |         3          | [Brief description]        |        N       | Group 1 N / Group 2 Y   |

---

## Explanation by Case

### Case A
[Explain Policy A Y/Y.]
[Explain Policy B Group 1 Y / Group 2 N.]

### Case B
[Explain Policy A N/N.]
[Explain Policy B Group 1 Y / Group 2 N.]

### Case C
[Explain Policy A Y/Y.]
[Explain Policy B Group 1 N / Group 2 Y.]

### Case D
[Explain Policy A N/N.]
[Explain Policy B Group 1 N / Group 2 Y.]

---

## Verification Table

| Case   | Policy A, Group 1 | Policy A, Group 2 | Policy B, Group 1 | Policy B, Group 2 |
| :----- | :---------------: | :---------------: | :---------------: | :---------------: |
| Case A |         Y         |         Y         |         Y         |         N         |
| Case B |         N         |         N         |         Y         |         N         |
| Case C |         Y         |         Y         |         N         |         Y         |
| Case D |         N         |         N         |         N         |         Y         |

---

## Why This Is Adversarial

[Why Policy A would likely be rated more ambiguous by a surface-level ambiguity judge, despite low group disagreement.]
[Why Policy B would likely be rated less ambiguous by a surface-level ambiguity judge, despite high group disagreement.]
```

When the example is later scaled up, extend (do not replace) the README: add rows to the cases table for new cases, list policy paraphrases under their parent policy, and note any added rater personas under the relevant group.

---

## Scaling options

After producing the minimal 2 / 4 / 2 set, offer the user the following scaling options. Do not apply them unprompted.

### 1. Paraphrase variation (more policies)

Generate additional wordings of Policy A and/or Policy B that preserve the same consensus or conflict axis. Append to `policies.json` with IDs like `consensus2`, `consensus3`, `conflict2`, ... Use this to test robustness to wording.

### 2. Cell variation (more cases)

Generate additional realistic cases within each of the four cells. Append to `cases.json` and add rows to the cases table in `README.md`. Cell counts need not be uniform — favor ecologically realistic frequencies over balanced quotas.

### 3. Rater paraphrase (more personas per group)

Generate additional personas within each existing group, varying role, vocabulary, and specific concerns while preserving the interpretive stance. Append to `longest_raters.json`. Use this to test whether the conflict pattern is robust across within-group stylistic variation.

### 4. Axis variation (new structures)

Choose a new consensus axis, a new conflict axis, or both, within the same domain. This produces a *new sibling subdir* (e.g., `adversarial_housing_displacement/` alongside `adversarial_housing_affordability/`) rather than extending the existing one. Each axis pairing is a separate adversarial structure.

---

# Compact Prompt for Another LLM

Use the following prompt when asking another LLM to generate examples.

```markdown
Generate an adversarial ambiguity example in the domain of [DOMAIN].

Follow the procedure in AGENTS.md. Produce a new subdirectory named after the domain (snake_case, prefixed with `adversarial_`, e.g. `adversarial_housing/`), as a sibling to `example-format/`, containing exactly four files:

1. `policies.json` — JSON array of `{candidate_id, text}` objects. Use IDs `consensus1` for Policy A and `conflict1` for Policy B.
2. `cases.json` — JSON array of four case-description strings, ordered Case A, B, C, D.
3. `longest_raters.json` — JSON array of two rater-persona description strings, ordered Group 1, Group 2.
4. `README.md` — markdown documenting the groups, the consensus and conflict axes, both policies, the four cases with cell assignments, per-case Y/N explanations, the verification table, and a short why-adversarial summary.

Create two groups representing plausible interpretive communities in the domain, differing along a real political, normative, institutional, or interpretive cleavage.

Create two policies:

1. Policy A should sound ambiguous (vague, value-laden language) but produce agreement between the groups on the sampled cases. It should track a consensus axis both groups classify the same way.
2. Policy B should sound concrete (explicit, operational, administrative, measurable-sounding language) but produce disagreement between the groups. It should track a conflict axis the groups interpret differently.

Create four realistic cases — not artificial edge cases — satisfying this matrix:

| Case   | Policy A, Group 1 | Policy A, Group 2 | Policy B, Group 1 | Policy B, Group 2 |
| :----- | :---------------: | :---------------: | :---------------: | :---------------: |
| Case A |         Y         |         Y         |         Y         |         N         |
| Case B |         N         |         N         |         Y         |         N         |
| Case C |         Y         |         Y         |         N         |         Y         |
| Case D |         N         |         N         |         N         |         Y         |

In `README.md`, explain each Y/N judgment, name the consensus and conflict axes, include the verification table, and add a short explanation of why this is adversarial to simple LLM-as-a-judge ambiguity ratings.

Default to the minimal 2 / 4 / 2 set described above. Do not scale up unless asked. After producing the set, surface the scaling options listed in AGENTS.md (paraphrase, cell, rater, axis variation) so the user can choose.

Make sure the cases are realistic for the domain and that both groups are represented charitably rather than as caricatures.
```

---

# Key Principle

The central trick is:

> Make surface ambiguity and operational disagreement come apart.

Policy A should look vague but behave concretely on the sampled cases.

Policy B should look concrete but import a contested social, political, or normative concept whose implementation divides the groups.

The cases are then generated as a cross-product of:

1. a **consensus dimension** that both groups classify the same way; and
2. a **conflict dimension** that the groups classify differently.
