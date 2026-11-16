"""Global official matching, false possessions and analysis-only D/component oracles."""
import numpy as np
from possession.learning.study import ScoreContext
from possession.phase5.metrics import diagnostics
from possession.phase6.decoding import DirectionalDecoder
from possession.learning.temporal import TemporalDecoder


def hard_head(context, p):
    label=np.where(p[:,:2].sum(1)>=.5,(p[:,1]>p[:,0]).astype(int),2)
    return context.decoded(np.eye(3)[label],TemporalDecoder({"kind":"argmax"},None))


def matched(context, hard):
    pm=context.pm; truth=pm.possession_spells(context.gt,context.runs)
    predicted=pm.possession_spells(hard,context.runs)
    overlap=pm.spell_overlaps(truth,predicted)
    return truth,predicted,pm.matched_gt_flags(overlap,len(truth),.5)


def measure(context, p, prior):
    decoder=DirectionalDecoder(context.view,prior,"hmm",.5)
    hard=context.decoded(p,decoder); pm=context.pm
    full=pm.evaluate(context.gt,hard,bootstrap_n=0)
    detail=diagnostics(context,hard); groups=detail["duration_groups"]
    minutes=sum(b-a for a,b in context.runs)/25/60
    gt=sum(groups[g]["gt"] for g in ("lt1","1to3"))
    recovered=sum(groups[g]["iou"]["0.5"]["matched_gt"] for g in ("lt1","1to3"))
    fp=sum(groups[g]["iou"]["0.5"]["fp"] for g in groups)
    return {"spellF1":float(pm.spell_scores(context.gt,hard,context.runs)["mean_f1"]),
            "macroF1":float(full["macro_f1"]),"long_recall":groups["ge10"]["iou"]["0.5"]["recall"],
            "short_recall":recovered/max(1,gt),"short_recovered":recovered,"short_gt":gt,
            "false_possessions":fp,"false_possessions_per_minute":fp/max(1e-9,minutes),"minutes":minutes,
            "fragmented_gt":detail["fragmented_gt"],"fused_predictions":detail["fused_predictions"],
            "gt_spells":detail["gt_spells"],"pred_spells":detail["pred_spells"],
            "duration_groups":groups,"decoded_HA":detail["decoded_HA"]}


def analyze(context,p,teacher_p,prior):
    pm=context.pm; decoder=DirectionalDecoder(context.view,prior,"hmm",.5)
    final=context.decoded(p,decoder)
    raw=hard_head(context,p); original_raw=hard_head(context,teacher_p)
    original=context.decoded(teacher_p,decoder)
    truth,_,new_flags=matched(context,final)
    teacher_flags=matched(context,original)[2]
    teacher_raw_flags=matched(context,original_raw)[2]
    own_raw_flags=matched(context,raw)[2]
    native_team=np.zeros(len(context.gt),np.int8); native_alive=np.zeros(len(context.gt),float)
    conditional=(p[:,1]>p[:,0]).astype(np.int8)
    for selected,mapping,output in context.parts:
        native_team[output]=conditional[selected][mapping]
        native_alive[output]=teacher_p[selected,:2].sum(1)[mapping]
    oracle=np.where(context.gt==2,2,native_team).astype(np.int8)
    oracle[context.gt==3]=3
    teacher_true_team=final.copy()
    valid=(context.gt<2)&(final<2); teacher_true_team[valid]=context.gt[valid]
    details=[]
    for j,(a,b,team) in enumerate(truth):
        old=int(context.gt[a-1]) if a else 3
        new=int(context.gt[b]) if b<len(context.gt) else 3
        if old==3 or new==3:kind="censored"
        elif old==2 and new==2:kind="D_D"
        elif old<2 and new<2:kind="HA_HA"
        else:kind="mixed_HA_D"
        details.append({"duration_s":(b-a)/25,"boundary_type":kind,"recovered":bool(new_flags[j]),
            "teacher_recovered":bool(teacher_flags[j]),"teacher_raw_recovered":bool(teacher_raw_flags[j]),
            "own_raw_recovered":bool(own_raw_flags[j]),
            "teacher_raw_spell_HMM_removed":bool(teacher_raw_flags[j] and not teacher_flags[j]),
            "no_teacher_raw_correct_frame":bool(not (original_raw[a:b]==team).any()),
            "frozen_dead_on_all_frames":bool((native_alive[a:b]<.5).all()),
            "frozen_dead_fraction":float((native_alive[a:b]<.5).mean())})
    return {"per_possession":details,
        "true_D_predicted_team_oracle_spellF1":float(pm.spell_scores(context.gt,oracle,context.runs)["mean_f1"]),
        "true_team_where_defined_predicted_D_oracle_spellF1":float(pm.spell_scores(context.gt,teacher_true_team,context.runs)["mean_f1"]),
        "note":"Analysis only. Frozen alive probabilities do not freeze hard D after joint HMM. Missing raw evidence is operational, not proof of absent latent information."}
