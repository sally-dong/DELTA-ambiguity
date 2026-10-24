# Adversarial Ambiguity Example: Australian Republic Referendum

> **Note on variant — aspiration/implementation.** This example uses the aspiration/implementation framing. The "policies" are constitutional-reform aspirations and the "cases" are implementations of those aspirations. Y means the group judges the implementation to be consistent with the aspiration; N means not consistent.
>
> **Note on variant — matrix.** This example intentionally departs from the standard adversarial matrix. In the standard pattern, Policy B produces strict Group 1 ≠ Group 2 disagreement on every case. Here, Policy B produces *coalitional cross-pressures*: Case A is rejected by both groups (for incompatible reasons) and Case D is accepted by both groups (also for different reasons), while Cases B and C show the standard split. This pattern is historically grounded — the 1999 Australian republic referendum's official "No" case explicitly drew support from both monarchists and direct-election republicans, who rejected the same model for incompatible reasons. The example remains adversarial because Policy B's surface concreteness still hides contested interpretation of "head of state," "Australian," and "republic"; the disagreement simply runs along a richer pattern than pure G1/G2 opposition.

## Groups

**Group 1: Monarchist / constitutional-continuity reviewers** — corresponds to `longest_raters.json[0–14]`.
They value the existing Westminster settlement, stability, reserve-power restraint, and a non-political ceremonial umpire. They can mark a republican implementation as consistent with the aspiration if, despite opposing it politically, they think it genuinely creates a stable, independent Australian head of state. The 15 personas span constitutional-monarchy advocacy spokespeople, senior constitutional silks, Australians for Constitutional Monarchy branch presidents, Westminster constitutional-law professors, retired official secretaries to Governors-General, conservative national-broadsheet columnists, retired Liberal senators on constitutional committees, retired state supreme-court justices, federation historians, ACM national committee members, traditionalist Anglican clergy, RSL veteran members, retired senior diplomats, regional civic leaders, and parliamentary committee secretaries.

**Group 2: Republican / popular-sovereignty reviewers** — corresponds to `longest_raters.json[15–29]`.
They want to end hereditary monarchy and create a democratically legitimate Australian head of state. They are suspicious of elite appointment, Prime Ministerial control, and "politicians' republic" models. The 15 personas span Australian Republic Movement activists, ARM national chairs, comparative republican constitutional scholars, senior Labor politicians, Greens democratic-reform advocates, Indigenous sovereignty scholars, civics-education organisation directors, citizens'-assembly advocates, ACTU constitutional-reform officials, young digital-democracy organisers, republican regional mayors, civics teachers, Westminster-republic comparative scholars, republican essayists, and historians of Australian national identity.

Y = the group thinks the implementation is consistent with the aspiration.
N = the group thinks the implementation is not consistent with the aspiration.

---

## Axes

**Consensus axis (tracked by Policy A):** Is the constitutional-settlement process fair, durable, and people-owned — featuring genuine public deliberation over the model and a public role in choosing the head of state — rather than a tactical procedural device or an elite-designed model offered to the public only as a final yes/no?

**Conflict axis (tracked by Policy B):** What counts as a "republic" and an "Australian head of state"?

| Hidden question                            | Group 1 reading (constitutional continuity)                                | Group 2 reading (popular sovereignty)                          |
| :----------------------------------------- | :------------------------------------------------------------------------- | :------------------------------------------------------------- |
| What counts as a "head of state"?          | A stable, ceremonial, non-political constitutional umpire.                 | A democratically legitimate national representative.           |
| What counts as "Australian"?               | Locally constituted and independent enough to preserve the system.         | Chosen by Australians, not inherited or elite-appointed.       |
| What counts as a "republic"?               | A risky institutional redesign that must preserve continuity if attempted. | A break with monarchy requiring popular democratic legitimacy. |
| What is wrong with the Howard-style model? | A flawed politicians' president, not a stable umpire.                      | A politicians' republic, not a people's republic.              |

---

## Policy A — `consensus1`–`consensus10`: seems ambiguous, actually concrete

> Australia should resolve the head-of-state question through a fair, durable, people-owned constitutional settlement, not a tactical stitch-up or symbolic half-measure.

The text above is `consensus1`; nine paraphrases (`consensus2`–`consensus10`) appear in `policies.json`. The family is full of ambiguity cues: "fair," "durable," "people-owned," "tactical stitch-up," "symbolic half-measure," and variants like "meaningful," "genuine," "stitch-up," "hollow gesture," "fait accompli." A surface-level ambiguity rater would mark each as highly vague. But on the sampled implementations both groups agree on every answer, because the cases differ on a concrete dimension: did the process feature genuine public deliberation and a public role in selecting the head of state?

## Policy B — `conflict1`–`conflict10`: seems concrete, actually ambiguous

> Australia should become a republic and have an Australian head of state.

The text above is `conflict1`; nine paraphrases (`conflict2`–`conflict10`) appear in `policies.json`. Each reads as a straightforward institutional prescription — "become a republic," "end the monarchy," "have an Australian head of state." A surface-level rater would mark each as relatively concrete. But the real ambiguity is in the contested terms mapped above. Different groups disagree about what kind of office "head of state" describes, what "Australian" requires, and what counts as a "republic" — producing the variant pattern described in the matrix note.

---

## Cases (implementations)

| Case   | `sampled_cases.json` index | Description                                                                                                  | Consensus axis | Conflict axis              |
| :----- | :----------------: | :----------------------------------------------------------------------------------------------------------- | :------------: | :------------------------- |
| Case A |         0          | Howard / 1999-style parliamentary-appointment republic; single model, no prior popular choice                |       N        | Group 1 N / Group 2 N      |
| Case B |         1          | Expert-council minimalist republic; President chosen by elite council, public gets only a final yes/no       |       N        | Group 1 Y / Group 2 N      |
| Case C |         2          | Direct-election popular republic; ceremonial President elected by national vote after public deliberation   |       Y        | Group 1 N / Group 2 Y      |
| Case D |         3          | Citizens' shortlist compromise republic; assembly nominates shortlist, public chooses, final referendum     |       Y        | Group 1 Y / Group 2 Y      |

---

## Explanation by Case

### Case A — Howard / 1999-style parliamentary-appointment republic
**Policy A: N/N.** Both groups agree this is not a fair, durable, people-owned settlement. Monarchists read it as a tactical device for forcing a flawed republic to a vote; republicans read it as a poisoned-chalice model that lets the public reject "the republic" by rejecting one unpopular design.
**Policy B: N/N (coalitional cross-pressure).** Monarchists reject it because it creates a politicians' president vulnerable to Prime Ministerial dominance — not a stable constitutional umpire. Republicans reject it because the people do not choose the President — not a genuine republican head of state. Both groups land on N for incompatible reasons. This is the "nobody wanted this implementation" row.

### Case B — Expert-council minimalist republic
**Policy A: N/N.** Both groups agree the process is not sufficiently people-owned. It is elite-designed, technocratic, and gives voters no meaningful choice among models.
**Policy B: Group 1 Y / Group 2 N.** Monarchists can accept it: if Australia is going to become a republic, this at least creates the right kind of head of state — ceremonial, stable, non-partisan, continuous with the Governor-General's role. Republicans reject it: it abolishes the Crown formally but does not give Australians a democratically legitimate head of state. This captures the monarchist reading of "head of state" as neutral umpire versus the republican reading of "Australian" as requiring popular democratic legitimacy.

### Case C — Direct-election popular republic
**Policy A: Y/Y.** Both groups agree this is a fair and people-owned settlement process, even if monarchists oppose the substantive outcome.
**Policy B: Group 1 N / Group 2 Y.** Monarchists read direct election as creating a rival democratic mandate that risks turning the head of state into a political actor, undermining the Westminster system. Republicans read it as the clearest implementation of the aspiration: Australia becomes a republic and Australians choose their own head of state. This is the classic constitutional-stability vs popular-sovereignty split.

### Case D — Citizens' shortlist compromise republic
**Policy A: Y/Y.** Both groups agree the process is fair, durable, and people-owned: public deliberation, a genuine model-design process, a public role in choosing the head of state, and a final referendum.
**Policy B: Y/Y (coalitional convergence).** Monarchists, although they still prefer the monarchy, can accept this as a stable, non-partisan, constitutionally constrained Australian head of state. Republicans accept it because it ends the monarchy and gives Australians a meaningful role. Both land on Y for different reasons. This is the broad-compromise row.

---

## Verification Table

| Case   | Policy A, Group 1 | Policy A, Group 2 | Policy B, Group 1 | Policy B, Group 2 |
| :----- | :---------------: | :---------------: | :---------------: | :---------------: |
| Case A |         N         |         N         |         N         |         N         |
| Case B |         N         |         N         |         Y         |         N         |
| Case C |         Y         |         Y         |         N         |         Y         |
| Case D |         Y         |         Y         |         Y         |         Y         |

*Variant matrix — see the matrix note at the top of this document. Cases B and C match the standard adversarial pattern; Cases A and D are coalitional cross-pressure (both N) and coalitional convergence (both Y) rows respectively.*

---

## Why This Is Adversarial

Policy A would likely be rated more ambiguous by a surface-level ambiguity judge: phrases like "fair," "durable," "people-owned," "tactical stitch-up," and "symbolic half-measure" are rhetorical and evaluative. Yet on these implementations both groups agree on every answer, because the implementations differ on a concrete process-legitimacy axis.

Policy B would likely be rated less ambiguous by a surface-level judge: "Australia should become a republic and have an Australian head of state" reads as a clean institutional prescription. Yet the implementations produce a coalitional cross-pressure pattern — Case A rejected by both, Case D accepted by both, Cases B and C splitting opposite ways — precisely because "republic," "Australian," and "head of state" each encode contested constitutional concepts. The historical anchor is the 1999 Australian republic referendum, where the official "No" case explicitly drew support from both monarchists and direct-election republicans for the same model, mirroring the Case A pattern shown here.
