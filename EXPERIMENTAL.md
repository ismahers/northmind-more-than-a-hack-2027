# Reviewed experiments and unchanged submission

The selected recipe remains `northmind_sequence`. The 17 November control-memory and learned-live pilots are experimental, and neither passed all predeclared promotion gates. The separate learned-live target CSV is unselected; the original delivery CSV was preserved. This source overlay does not select, deploy or submit it.

## Public external results

The learned-live pilot improved the weighted SpellF1 objective by +0.05628 in fold 0 and +0.09142 in fold 6. The objective is 0.50 clean + 0.25 missing-ball 5 s + 0.25 missing-ball 15 s. It is not a worst-case score or an occlusion-only measure.

The clean short-possession recall guard failed in fold 0 (20/142 to 19/142). Short recall in its 5 s occlusion case also fell (14/142 to 9/142). Some occluded cases have more false H/A transitions despite better overall SpellF1. The other secondary guards apply to clean only. These tradeoffs are not grounds to loosen thresholds after seeing results. See [the full result](docs/experiments/LIVE_BALL_V1_RESULT_2026-11-17.md), [the preregistered protocol](docs/experiments/LIVE_BALL_V1_2026-11-17.md) and [the reviewed public aggregate](docs/public-results/2026-11-17/live-ball-v1.json).

The external folds were previously inspected IDSSE validation folds, with tactical priors neutralized. This does not validate the complete target ensemble with automatic Smart Tagging. No target GT, target scoring or video comparison was used. The memory pilot also remains unpromoted; its results and protocol are published separately.

## Verification and restoration limits

Recorded synthetic checks passed: 39 for control memory, 39 for learned-live validation and 25 for candidate preparation. Candidate preparation verified exact legacy-baseline labels and original CSV bytes, exact recovery of cached 10 Hz alive anchors, and two byte-identical generations from cold geometry. Only sanitized boolean status is public. The target CSV, native inputs, caches, logs and private provenance are not distributed.

The own learned 2x2 prior is supplied in `configs/live_ball_prior_v1.json`: the predeclared equal average of fold 0/6 transition probabilities. It requires no new model weights. The public model capsule and its eight original checkpoints are unchanged. Target inference uses cached float32 alive probabilities; external validation averaged in float64. Exact anchor reconstruction proves cache recovery, not equal precision or equal end-to-end performance.

Successful local checks do not establish cold reproducibility from this checkout. External raw inputs and historical experiment registrations/artifacts are not bundled. Sealed experiment checks may require restoration of their exact historical source snapshot; later launcher changes can invalidate their dependency hashes. The new candidate has not been replayed from another root or a fresh environment. The existing selected-recipe installation instructions and model-capsule provenance remain applicable to that recipe.

## Revised English technical report

`Northmind.md` and `docs/submission/Northmind.en.md` contain the revised English report source. It describes the unchanged selected recipe and clarifies its missing-observation D fallback. `PDF-BUILD.md` and `docs/submission/PDF.md` document the build. The revised PDF was rendered as four A4 pages and all pages were visually checked. This source overlay publishes source only: the historical Release PDF binary has not been replaced, and the old download link identifies that historical artifact.

`SOURCE-MANIFEST.json` records LF-normalized UTF-8 source bytes for the proposed public tree while preserving unaffected entries. It is an inventory, not a new cold-replay certificate. No restricted target output is included.

## Applying this source overlay

The archive contains 34 explicitly reviewed paths, intended as an overlay on public commit `8e5e3756d8d0f5554f50f1b8d775ccdcb5ed1448`. Unaffected manifest entries refer to files already present in that base checkout. GitHub issue and blob creation were rejected with HTTP 403 (resource not accessible by integration); no issue, branch or pull request was created, and main was not changed.
