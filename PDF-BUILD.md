# English technical report

The submission document is `Northmind.pdf`, in English, four A4 pages.
The official `SUBMISSION.md` accepts PDF or Markdown and does not explicitly
prescribe the document language. English is the team's chosen submission language.

Source: `docs/submission/Northmind.en.md`. Build with Python and ReportLab 4.4.9:

```text
python -m pip install reportlab==4.4.9
python scripts/build_technical_pdf.py
```

Output: `output/pdf/Northmind.pdf`. The builder reads public documentation only.
It does not load match data, model weights or predictions. Fonts are embedded
when Calibri/Arial are available on Windows. Regeneration on another platform
can change pagination and must be checked visually; the four-page guard rejects
overflow rather than silently producing an extra page.

The delivered PDF was rendered and all four pages visually checked on Windows.
It has selectable text, three clickable source links and no clipped content.
English source clarifies that the bridged sequence is an alternative, while the
saved fallback CSV is geometric. No algorithm or prediction has been changed.

Review on 17 November also clarified the observation-missing D fallback and
separated the new external pilot results from the unchanged submitted recipe.
The revised document was regenerated and all four pages visually checked again.
The reviewed final PDF is included at the repository root as `Northmind.pdf`.
`PUBLIC-DELIVERY-MANIFEST.json` records its exact binary size and SHA-256.
The `submission-v1` Release PDF is historical and has not been replaced; use
the canonical download below for the current document.

Download: [Northmind.pdf](https://github.com/ismahers/northmind-more-than-a-hack-2027/raw/refs/heads/main/Northmind.pdf)
