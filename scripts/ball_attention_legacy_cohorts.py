"""Post-seal diagnosis on the original 209/133 cohort, with cached cases only."""
from possession.temporal_probe.common import read_json
from possession.night.common import atomic_json
from possession.external_data.schema import file_hash
from possession.ball_attention.common import ROOT,DEST,PARENT,verify
from possession.ball_attention.experiment import assert_sealed,cohorts


def main():
    record = read_json(DEST/"registration.json");digest = file_hash(DEST/"registration.json")
    assert_sealed(record,digest)
    checks = verify(record)
    if not read_json(DEST/"verification.json")["passed"] or not all(checks.values()):
        raise ValueError("Unverified pilot")
    results = {};refs = {}
    for part in ("validation","test"):
        results[part] = {}
        for fold in (0,6):
            legacy_path = PARENT/f"cases/{part}/fold-{fold}.json"
            legacy = read_json(legacy_path);refs[str(legacy_path.relative_to(ROOT))] = file_hash(legacy_path)
            result = {}
            for arm in ("original","ball","capacity"):
                path = DEST/f"cases/{part}/fold-{fold}/{arm}/clean.json"
                value = read_json(path);refs[str(path.relative_to(ROOT))] = file_hash(path)
                result[arm] = cohorts(legacy,value)
            results[part][str(fold)] = result
    atomic_json(DEST/"legacy-cohorts.json",{"results":results,"references":refs,
        "pilot_summary_sha256":file_hash(DEST/"summary.json"),"selection_seal_sha256":file_hash(DEST/"selection-seal.json"),
        "script_sha256":file_hash(__file__),"ancestry_checks_passed":all(checks.values()),"ancestry_checks_count":len(checks),
        "new_training":0,"new_predictions":0,"selection_changed":False,
        "scope":"Original local8 HA-discrimination cohorts, original GT spell order and duration/boundary alignment required. Diagnostic only after selections. Not a paired architectural effect versus local8, which has a different downstream head."})
    return 0


if __name__=="__main__":
    raise SystemExit(main())
