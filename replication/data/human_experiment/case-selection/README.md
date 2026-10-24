# Case selection

The 20 campus-speech cases in `../cases.json` were written by the authors,
paraphrasing real incidents drawn from three databases maintained by the
Foundation for Individual Rights and Expression (FIRE):

- **Campus Deplatforming Database** — <https://www.thefire.org/research-learn/campus-deplatforming-database>
  (formerly the Campus Disinvitation Database)
- **Scholars Under Fire Database** — <https://www.thefire.org/research-learn/scholars-under-fire>
- **Students Under Fire Database** — <https://www.thefire.org/research-learn/students-under-fire>

The versions we worked from were exported 2025-04-29 and 2025-05-14. The
Campus Deplatforming database was obtained from FIRE on request, as no export
function is offered.

**The databases themselves are not redistributed here.** Only our 20 rewritten
cases are published, in `../cases.json`. Where a case reproduces wording from
the source, it is a direct quotation of the speech at issue in the incident —
that quotation is the substance of the case — rather than FIRE's own
description of it.

`fire-cases.py` is the script used to browse the databases while shortlisting
candidate incidents: it keeps incidents whose `controversy_explanation`
contains a direct quote, breaks them out by topic, and prints samples for
reading. It does not mechanically generate `../cases.json` — the final 20 cases
were chosen and reworded by hand. To run it, download the three databases from
the URLs above and place the CSVs in a `fire-databases/` subdirectory here (or
repoint `DATA_DIR` at the top of the script).

Please cite FIRE directly for the underlying incident data.
