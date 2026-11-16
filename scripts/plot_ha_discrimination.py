"""Plot only aggregate diagnostic curves, never raw match assets or logs."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root=Path(__file__).resolve().parents[1]/"evaluation/reports/external/ha-discrimination-v1"
    summary=json.loads((root/"diagnostic-summary.json").read_text())
    fig,axes=plt.subplots(2,2,figsize=(10,6.5),sharex=True,sharey=True)
    for column,row in enumerate(summary["test"]):
        for axis_row,side in enumerate(("entry","exit")):
            ax=axes[axis_row,column]
            for kind,label,color in (("HA_HA","Extremos H↔A", "#6554c0"),("mixed_HA_D","Extremos mixtos H/A–D", "#008577")):
                values=[]
                for key,value in row["curves"].items():
                    group,edge,offset=key.split("/")
                    if group==kind and edge==side:values.append((float(offset),value["probability"]["mean"]))
                values.sort()
                ax.plot([x for x,_ in values],[y for _,y in values],"o-",color=color,label=label,markersize=4)
            ax.axhline(.5,color="#777777",ls="--",lw=.8)
            ax.axvline(0,color="#999999",ls=":",lw=.8)
            ax.set_ylim(0,1);ax.grid(alpha=.15)
            ax.set_title(f"Fold {row['fold']} · {'entrada' if side=='entry' else 'salida'}")
            ax.set_xlabel("Segundos respecto a la frontera GT")
    axes[0,0].legend(fontsize=8,loc="upper left")
    fig.supylabel("P(equipo de la posesión corta | vivo), media",fontsize=10,x=.012)
    fig.suptitle("Emisiones existentes de la MLP local de 8 épocas · test descriptivo",fontsize=13)
    fig.text(.5,.015,"Cada curva conserva el equipo de la posesión corta también después de su salida. No es una curva causal ni un sistema nuevo.",ha="center",fontsize=8)
    fig.tight_layout(rect=(.035,.04,1,.95))
    folder=root/"figures";folder.mkdir(exist_ok=True)
    fig.savefig(folder/"transition_probabilities.png",dpi=160)
    fig.savefig(folder/"transition_probabilities.svg")
    plt.close(fig)
    return 0
