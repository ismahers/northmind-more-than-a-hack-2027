"""Explicit public roots; never discover or inspect hackathon asset locations."""
from pathlib import Path
import json
from dataclasses import fields
import numpy as np
from possession.external_data.features import Features, build_features
from possession.external_data.schema import Match
from possession.night.cache import PublicStore
from possession.night.common import atomic_json
from scripts.external_data import ROOT, REPORTS
from scripts.train_e0 import manifest
from .features import harmonize

DEST = REPORTS / "phase7-v1"


def records():
    data, _ = manifest("idsse-skillcorner-phase3")
    for key, record in data["artifacts"].items():
        if not key.startswith(("idsse:", "skillcorner:")):
            raise ValueError("Nonpublic provider rejected")
        if not Path(record["path"]).resolve().is_relative_to((ROOT/"data/external").resolve()):
            raise ValueError("Artifact outside explicit public external root")
    return data["artifacts"]


class Store(PublicStore):
    def __init__(self):
        super().__init__(records())
        self.legacy = PublicStore({k:v for k,v in self.records.items() if k.startswith("idsse:")})
        self.common = {}

    def view(self, key):
        # Existing IDSSE cache is read only, and never populated with other keys.
        if key.startswith("idsse:"):
            return self.legacy.view(key)
        return build_features(self.match(key))

    def harmonized(self, key):
        if key not in self.common:
            folder = DEST/"feature_cache"/key.replace(":", "_")
            marker = folder/"complete.json"
            if not marker.exists():
                v = harmonize(self.view(key)); folder.mkdir(parents=True, exist_ok=True)
                for f in fields(Features):
                    np.save(folder/(f.name+".npy"), getattr(v,f.name), allow_pickle=False)
                atomic_json(marker, {"artifact":self.records[key]["sha256"], "version":"common-xy-v1"})
            self.common[key] = Features(**{f.name:np.load(folder/(f.name+".npy"),mmap_mode="c",allow_pickle=False)
                                         for f in fields(Features)})
        return self.common[key]


def protocol():
    spec=json.loads((ROOT/"configs/phase7_protocol.json").read_text())
    keys=sorted(k for k in records() if k.startswith("skillcorner:"))
    spec["skillcorner_development_keys"],spec["skillcorner_confirmation_keys"]=keys[:4],keys[4:]
    path=DEST/"protocol.json"
    if path.exists() and json.loads(path.read_text())!=spec:
        raise ValueError("Frozen Phase7 protocol differs; create a new version")
    if not path.exists():atomic_json(path,spec)
    return spec
