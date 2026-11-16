"""Post-selection public error analysis; never changes models, selectors or decoders."""
import argparse
import numpy as np
from possession.external_data.schema import file_hash
from possession.night.common import atomic_json
from possession.duration_balanced.common import destination,read_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--source",choices=("isma","phase8"),default="isma")
    args=parser.parse_args();dest=destination(args.source)
    if not (dest/"all_rows.json").is_file():return 2
    rows=read_json(dest/"all_rows.json"); result={}
    for case in ("clean","missing40_5s","missing40_15s"):
        result[case]={}
        for arm in ("original","duration_cap2","duration_cap3"):
            all_details=[];counts={};oracles=[]
            for fold in (0,6):
                current=next(r for r in rows if (r["fold"],r["arm"],r["scenario"])==(fold,arm,case))
                baseline=next(r for r in rows if (r["fold"],r["arm"],r["scenario"])==(fold,"original",case))
                details=current["analysis"]["per_possession"];old=baseline["analysis"]["per_possession"]
                assert len(details)==len(old)
                for a,b in zip(details,old):
                    assert a["duration_s"]==b["duration_s"] and a["boundary_type"]==b["boundary_type"]
                    all_details.append(dict(a,original_recovered=b["recovered"]))
                oracles.append({k:v for k,v in current["analysis"].items() if "oracle_spellF1" in k})
            short=[d for d in all_details if d["duration_s"]<3]
            for kind in ("HA_HA","D_D","mixed_HA_D","censored"):
                group=[d for d in short if d["boundary_type"]==kind]
                counts[kind]={"gt":len(group),"original":sum(d["original_recovered"] for d in group),
                    "current":sum(d["recovered"] for d in group),
                    "gained_vs_original":sum(d["recovered"] and not d["original_recovered"] for d in group),
                    "lost_vs_original":sum(d["original_recovered"] and not d["recovered"] for d in group)}
            subsets={}
            for key in ("teacher_raw_spell_HMM_removed","no_teacher_raw_correct_frame","frozen_dead_on_all_frames"):
                group=[d for d in short if d[key]]
                subsets[key]={"gt":len(group),"original":sum(d["original_recovered"] for d in group),
                              "current":sum(d["recovered"] for d in group)}
            result[case][arm]={"types":counts,"operational_subsets":subsets,
                              "oracles":{k:float(np.mean([o[k] for o in oracles])) for k in oracles[0]}}
    result.update(post_selection_only=True,source=args.source,source_sha256=file_hash(__file__),
                  evaluated_rows_sha256=file_hash(dest/"all_rows.json"))
    atomic_json(dest/"analysis_summary.json",result)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    colors={"original":"#444444","duration_cap2":"#1565c0","duration_cap3":"#e08020"}
    for col,fold in enumerate((0,6)):
        for arm in colors:
            value=read_json(dest/f"training/fold-{fold}/{arm}/complete.json")
            h=value["history"]
            axes[col].plot([r["epoch"] for r in h],[r["loss"] for r in h],label=arm,color=colors[arm])
        axes[col].set(title=f"Fold {fold} — own weighted objective",xlabel="Epoch",ylabel="Training loss")
        axes[col].grid(alpha=.2);axes[col].legend(fontsize=8)
    fig.tight_layout();fig.savefig(dest/"learning_curves.png",dpi=150);plt.close(fig)
    return 0
