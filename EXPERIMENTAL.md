# Reviewed experiments and unchanged submission

The selected recipe remains `northmind_sequence`. The 17 November control-memory and learned-live pilots are experimental, and neither passed all predeclared promotion gates. The separate learned-live target CSV is unselected; the original delivery CSV was preserved. This public package does not select, deploy or submit it.

## Public external results

The learned-live pilot improved the weighted SpellF1 objective by +0.05628 in fold 0 and +0.09142 in fold 6. The objective is 0.50 clean + 0.25 missing-ball 5 s + 0.25 missing-ball 15 s. It is not a worst-case score or an occlusion-only measure.

The clean short-possession recall guard failed in fold 0 (20/142 to 19/142). Short recall in its 5 s occlusion case also fell (14/142 to 9/142). Some occluded cases have more false H/A transitions despite better overall SpellF1. The other secondary guards apply to clean only. These tradeoffs are not grounds to loosen thresholds after seeing results. See [the full result](docs/experiments/LIVE_BALL_V1_RESULT_2026-11-17.md), [the preregistered protocol](docs/experiments/LIVE_BALL_V1_2026-11-17.md) and [the reviewed public aggregate](docs/public-results/2026-11-17/live-ball-v1.json).

The external folds were previously inspected IDSSE validation folds, with tactical priors neutralized. This does not validate the complete target ensemble with automatic Smart Tagging. No target GT, target scoring or video comparison was used. The memory pilot also remains unpromoted; its results and protocol are published separately.

## Verification and restoration limits

Recorded synthetic checks passed: 39 for control memory, 39 for learned-live validation and 25 for candidate preparation. Candidate preparation verified exact legacy-baseline labels and original CSV bytes, exact recovery of cached 10 Hz alive anchors, and two byte-identical generations from cold geometry. Only sanitized boolean status is public. The target CSV, native inputs, caches, logs and private provenance are not distributed.

The own learned 2x2 prior is supplied in `configs/live_ball_prior_v1.json`: the predeclared equal average of fold 0/6 transition probabilities. It requires no new model weights. The public model capsule and its eight original checkpoints are unchanged. Target inference uses cached float32 alive probabilities; external validation averaged in float64. Exact anchor reconstruction proves cache recovery, not equal precision or equal end-to-end performance.

Successful local checks do not establish cold reproducibility from this checkout. External raw inputs and historical experiment registrations/artifacts are not bundled. Sealed experiment checks may require restoration of their exact historical source snapshot; later launcher changes can invalidate their dependency hashes. The new candidate has not been replayed from another root or a fresh environment. The existing selected-recipe installation instructions and model-capsule provenance remain applicable to that recipe.

## Revised English technical report

`Northmind.md` and `docs/submission/Northmind.en.md` contain the revised English report source. It describes the unchanged selected recipe and clarifies its missing-observation D fallback. `PDF-BUILD.md` and `docs/submission/PDF.md` document the build. The revised PDF was rendered as four A4 pages and all pages were visually checked. The reviewed final binary is now included at the repository root: [Northmind.pdf](https://github.com/ismahers/northmind-more-than-a-hack-2027/raw/refs/heads/main/Northmind.pdf). Its exact bytes and SHA-256 are recorded in `PUBLIC-DELIVERY-MANIFEST.json`. The PDF in the historical `submission-v1` Release has not been replaced; use the current root PDF.

`SOURCE-MANIFEST.json` records LF-normalized UTF-8 source bytes for this public tree while preserving unaffected entries. The binary PDF is inventoried separately in `PUBLIC-DELIVERY-MANIFEST.json`. These are inventories, not new cold-replay certificates. No restricted target output is included.

## Final public package

The reviewed experiment sources, protocols and public aggregates are included in this repository. The final delivery adds the reviewed PDF without changing the selected recipe, the eight original checkpoints or their authoritative public model capsule in `PUBLIC-MODEL-MANIFEST.json`. The selected CSV remains a separate private form attachment; no target assets or predictions are distributed here.
