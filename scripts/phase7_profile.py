"""Isolated, serial synthetic-batch cost profiling; never load official data."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from possession.phase7.model import SpatialE0
from possession.phase7.public import DEST
from possession.learning.training import seed_everything,memory_usage
from possession.night.common import atomic_json


def worker(arm):
    torch.set_num_threads(4);seed_everything(2027)
    if arm=="capacity":
        from possession.phase7_capacity.model import CapacityE0
        model=CapacityE0(True)
    else:
        model=SpatialE0(arm,True)
    x=torch.randn(4,286,2,18,6);x[...,4:6]=0
    mask=torch.zeros(4,286,2,18,dtype=torch.bool);mask[...,:11]=True
    g=torch.randn(4,286,7);g[:, :, [2,4,5]]=0;g[:,:,3]=1;g[:,:,6]=0
    args=(x,mask,g,torch.ones(4,286,dtype=torch.bool))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    before=memory_usage(torch.device("cpu"))["process_peak_rss_bytes"]
    steps=[]
    for j in range(5):
        start=time.perf_counter();model.train();optimizer.zero_grad(set_to_none=True)
        out=model(*args);loss=sum(out[k].square().mean() for k in ("alive_logit","team_logit","boundary_logit"))
        loss.backward();optimizer.step()
        if j>=2:steps.append(time.perf_counter()-start)
    peak=memory_usage(torch.device("cpu"))["process_peak_rss_bytes"]
    times=[];model.eval()
    with torch.no_grad():
        for j in range(4):
            start=time.perf_counter();model(*args)
            if j:times.append(time.perf_counter()-start)
    atomic_json(DEST/"profile"/(arm+".json"),{"arm":arm,"synthetic_cost_only":True,"batch_shape":[4,286,2,18,6],
        "available_players_per_team":11,"training_step_s_mean":float(np.mean(steps)),"inference_batch_s_mean":float(np.mean(times)),
        "process_peak_rss_before_bytes":before,"process_peak_rss_training_bytes":peak,"peak_rss_increment_bytes":peak-before,
        "parameters":sum(p.numel() for p in model.parameters()),"threads":4,"torch":torch.__version__})


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--worker",choices=("harmonized","capacity","set_transformer"));args=parser.parse_args()
    if args.worker:worker(args.worker);return 0
    folder=DEST/"profile";folder.mkdir(parents=True,exist_ok=True)
    for arm in ("harmonized","capacity","set_transformer"):
        if (folder/(arm+".json")).exists():continue
        with (folder/(arm+".log")).open("wb") as log:
            result=subprocess.run([sys.executable,"-m","scripts.phase7_profile","--worker",arm],stdout=log,stderr=log,stdin=subprocess.DEVNULL)
        if result.returncode or not (folder/(arm+".json")).exists():return 1
    return 0


if __name__=="__main__":raise SystemExit(main())
