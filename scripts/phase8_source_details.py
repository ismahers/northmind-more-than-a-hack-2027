"""Pinned public source/configuration reads and weight HEAD requests only."""
import hashlib
import json
import ssl
import urllib.request
import urllib.error
import certifi
from scripts.phase8_sources import DEST
from possession.night.common import atomic_json


def main():
    tls=ssl.create_default_context(cafile=certifi.where());folder=DEST/"details";folder.mkdir(parents=True,exist_ok=True)
    jobs=[]
    model=json.loads((DEST/"hf_model.txt").read_text());pin=model["sha"]
    for path in ("round3/FINAL.json","round3/final/args.json","round3/final/best.json","round3/final/best_team.json","round3/session1/best_team.json"):
        jobs.append(("hf_"+path.replace("/","_"),f"https://huggingface.co/Youssef16Ahmed/nova-pathcrf-finetuned/raw/{pin}/{path}","GET"))
    jobs.append(("hf_weights_head",f"https://huggingface.co/Youssef16Ahmed/nova-pathcrf-finetuned/resolve/{pin}/round3/final/model/state_dict_best_team.pt","HEAD"))
    repositories={
        "original":("hyunsungkim-ds/pathcrf",["saved/100/args.json","saved/110/args.json","saved/120/args.json","saved/130/args.json","saved/140/args.json","models/dynamic_sparse_crf.py","models/set_lstm.py","models/utils.py","dataset.py","inference.py","train.py","datatools/config.py","applications/team_poss.py","LICENSE"]),
        "tridet":("dingfengshi/TriDet",["libs/modeling/meta_archs.py","libs/modeling/blocks.py","libs/modeling/backbones.py","libs/modeling/losses.py","libs/utils/nms.py","libs/utils/setup.py","requirements.txt","LICENSE"]),
        "actionformer":("happyharrycn/actionformer_release",["libs/modeling/meta_archs.py","libs/modeling/backbones.py","libs/modeling/losses.py","libs/utils/nms.py","libs/utils/setup.py","INSTALL.md","LICENSE"])
    }
    for name,(repo,paths) in repositories.items():
        tree=json.loads((DEST/(name+"_tree.txt")).read_text());valid={x["path"] for x in tree["tree"]}
        for path in paths:
            if path in valid:jobs.append((name+"_"+path.replace("/","_"),f"https://raw.githubusercontent.com/{repo}/{tree['sha']}/{path}","GET"))
    rows=[]
    for name,url,method in jobs:
        path=folder/(name+".txt")
        if path.exists():rows.append({"name":name,"url":url,"status":"cached","sha256":hashlib.sha256(path.read_bytes()).hexdigest()});continue
        try:
            request=urllib.request.Request(url,method=method,headers={"User-Agent":"Northmind-public-source-study/1.0"})
            with urllib.request.urlopen(request,timeout=30,context=tls) as response:
                if method=="HEAD":
                    row={"name":name,"url":url,"status":response.status,"content_length":response.headers.get("Content-Length"),"body_read":False}
                else:
                    content=response.read(200001)
                    if len(content)>200000:raise ValueError("Text source size limit")
                    path.write_bytes(content);row={"name":name,"url":url,"status":response.status,"bytes":len(content),"sha256":hashlib.sha256(content).hexdigest()}
                rows.append(row)
        except urllib.error.HTTPError as error:rows.append({"name":name,"url":url,"status":error.code})
        except BaseException as error:rows.append({"name":name,"url":url,"status":"network_error","type":type(error).__name__})
    atomic_json(folder/"manifest.json",{"files":rows,"weights_downloaded":False,"untrusted_code_executed":False,"weights_deserialized":False})
    return 0
