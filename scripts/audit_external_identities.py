"""Inspect only public-source match metadata to check physical-match overlap."""

import json
import xml.etree.ElementTree as ET

from scripts.external_data import EXTERNAL, REPORTS, write_json
from possession.external_data.idsse import tag


def main():
    idsse, skillcorner = [], []
    for path in sorted((EXTERNAL / "raw/idsse").glob("*matchinformation*.xml")):
        root = ET.parse(path).getroot()
        general = next(x for x in root.iter() if tag(x) == "General")
        idsse.append({"file": path.name, "identity": {k: v for k, v in general.attrib.items()
                      if any(term in k.lower() for term in ("date", "time", "team", "matchid"))}})
    for path in sorted((EXTERNAL / "raw/skillcorner").glob("*_match.json")):
        meta = json.loads(path.read_text())
        dates = {k: v for k, v in meta.items() if any(term in k.lower() for term in ("date", "time"))
                 and isinstance(v, (str, int, float))}
        skillcorner.append({"file": path.name, "id": meta["id"], "date_fields": dates,
            "home": meta["home_team"].get("name"), "away": meta["away_team"].get("name")})
    a_dates = {r["identity"]["KickoffTime"][:10] for r in idsse}
    b_dates = {r["date_fields"]["date_time"][:10] for r in skillcorner}
    write_json(REPORTS / "phase3/source_identities.json", {"idsse": idsse, "skillcorner": skillcorner,
        "crossprovider_match_overlap_excluded_by_date": bool(idsse and skillcorner and not a_dates & b_dates),
        "source": "public metadata only; exact native identifiers retained; review dates and club names across providers"})
    return 0
