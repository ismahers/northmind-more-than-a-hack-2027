"""Read public model/repository metadata only; never download/deserialize weights."""
import json
from pathlib import Path
import urllib.request
import urllib.error
import ssl
import certifi
from possession.night.common import atomic_json
from scripts.external_data import REPORTS

DEST=REPORTS/"phase8-source-research"
URLS={
    "hf_model":"https://huggingface.co/api/models/Youssef16Ahmed/nova-pathcrf-finetuned?blobs=true",
    "hf_tree":"https://huggingface.co/api/models/Youssef16Ahmed/nova-pathcrf-finetuned/tree/main?recursive=true&expand=false",
    "hf_card":"https://huggingface.co/Youssef16Ahmed/nova-pathcrf-finetuned/raw/main/README.md",
    "original_tree":"https://api.github.com/repos/hyunsungkim-ds/pathcrf/git/trees/main?recursive=1",
    "tridet_tree":"https://api.github.com/repos/dingfengshi/TriDet/git/trees/master?recursive=1",
    "actionformer_tree":"https://api.github.com/repos/happyharrycn/actionformer_release/git/trees/main?recursive=1"
}


def main():
    DEST.mkdir(parents=True,exist_ok=True);rows=[]
    for name,url in URLS.items():
        path=DEST/(name+".txt")
        if path.exists():rows.append({"name":name,"url":url,"status":"cached_public_response"});continue
        try:
            request=urllib.request.Request(url,headers={"User-Agent":"Northmind-public-metadata-research/1.0"})
            with urllib.request.urlopen(request,timeout=25,context=ssl.create_default_context(cafile=certifi.where())) as response:
                content=response.read(2_000_001)
                if len(content)>2_000_000:raise ValueError("Metadata size limit")
                path.write_bytes(content)
                rows.append({"name":name,"url":url,"status":response.status,"bytes":len(content)})
        except urllib.error.HTTPError as error:rows.append({"name":name,"url":url,"status":error.code})
        except BaseException as error:rows.append({"name":name,"url":url,"status":"network_error","type":type(error).__name__,"reason_type":type(getattr(error,"reason",None)).__name__})
    atomic_json(DEST/"requests.json",{"requests":rows,"weights_downloaded":False,"deserialization":False})
    return 0
